"""From-scratch PPO training, validation selection and independent test decoding."""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from metro.envs.metro import MLine
from .env import GJFEnv, PreferenceState, normalize_weights, load_bundle, ROOT
from .models import ActorCritic, sparse_targets, attach_legal_gradient
from .features import legal_zscore


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_randomness():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    line_counter = MLine.tot_seq
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)
        MLine.tot_seq = line_counter


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def code_identity():
    files = [ROOT / "train.py", ROOT / "requirements.txt"]
    for folder in ("ours", "metro", "khrylib"):
        files.extend(sorted((ROOT / folder).rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_env(settings, weights):
    return GJFEnv(weights, city=settings["city"], initial_lines=settings["initial_lines"],
                  budget=settings["budget"], max_lines=settings["max_lines"],
                  min_stations=settings["min_stations"])


def scheduled_weights(target, profile, rollout, iterations, power):
    if profile != "commute":
        return target.copy()
    sharp = target ** power
    sharp /= sharp.sum()
    transition = max(1, math.ceil(iterations * .4))
    blend = min(1., rollout / transition)
    return normalize_weights((1. - blend) * sharp + blend * target)


def gae(rewards, values, next_values, done, lam):
    rewards, values, next_values = map(lambda x: np.asarray(x, dtype=np.float64),
                                      (rewards, values, next_values))
    advantages = np.zeros_like(rewards)
    future = np.zeros_like(rewards[0])
    for index in reversed(range(len(rewards))):
        continuation = 1. - float(done[index])
        delta = rewards[index] + continuation * next_values[index] - values[index]
        future = delta + lam * continuation * future
        advantages[index] = future
    return advantages, advantages + values


@torch.no_grad()
def decode(model, settings, weights, seed, deterministic, mode):
    seed_all(seed)
    env = make_env(settings, weights)
    state = env.reset(weights, mode)
    model.eval()
    while not env.done:
        dist, _ = model([state])
        action = int(dist.logits.argmax(-1).item()) if deterministic else int(dist.sample().item())
        state, _, _, _ = env.step(action)
    plan = env.plan()
    plan.update({"seed": int(seed), "deterministic": bool(deterministic),
                 "decoder_mode": int(mode)})
    return plan


def validation(model, settings, weights, seeds, mode):
    with preserve_randomness():
        plans = [decode(model, settings, weights, seeds[0], True, mode)]
        plans += [decode(model, settings, weights, seed, False, mode) for seed in seeds[1:]]
    sampled = np.asarray([plan["utility"] for plan in plans[1:]])
    best = max(plans, key=lambda plan: plan["utility"])
    result = {"deterministic": plans[0]["utility"], "best": best["utility"],
        "selection": best["utility"] + 1e-4 * plans[0]["utility"],
        "sample_distribution": {"min": float(sampled.min()), "median": float(np.median(sampled)),
                                "max": float(sampled.max()), "std": float(sampled.std())},
        "best_raw_GJF": best["raw_GJF"], "deterministic_raw_GJF": plans[0]["raw_GJF"],
        "plans": plans, "decoder_mode": int(mode)}
    return result


def collect(model, env, state, steps, train_weights, mode, episode_id, partial_return):
    records, completed = [], []
    model.eval()
    env.decoder_mode = mode
    state = env.observation()
    for _ in range(steps):
        with torch.no_grad():
            distribution, value = model([state])
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        next_state, vector, done, _ = env.step(int(action.item()))
        scalar = float(np.dot(state.preference_w.astype(np.float64), vector))
        with torch.no_grad():
            next_value = 0. if done else float(model([next_state])[1].item())
        records.append({"state": state.clone(), "action": int(action.item()),
            "old_log_prob": float(log_prob.item()), "value": float(value.item()),
            "next_value": next_value, "reward": scalar, "done": done, "episode": episode_id,
            "reward_vector": vector.copy()})
        partial_return += scalar
        state = next_state
        if done:
            plan = env.plan()
            np.testing.assert_allclose(partial_return, plan["utility"], rtol=1e-6, atol=1e-7)
            completed.append({"episode": episode_id, "plan": plan})
            episode_id += 1
            partial_return = 0.
            state = env.reset(train_weights, mode)
    return records, completed, state, episode_id, partial_return


def update(model, optimizer, records, config, rng, tail, incumbent, completed):
    device = next(model.parameters()).device
    advantages, returns = gae([r["reward"] for r in records], [r["value"] for r in records],
        [r["next_value"] for r in records], [r["done"] for r in records], config["tau"])
    peak = max(completed, key=lambda r: r["plan"]["utility"]) if completed else None
    promote = bool(tail and peak is not None and peak["plan"]["utility"] > incumbent + 1e-6)
    critic_only = tail and not promote
    if promote:
        advantages = np.asarray([float(r["episode"] == peak["episode"]) for r in records])
    elif not critic_only:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    metrics = []
    batch_size = config["mini_batch_size"]
    model.train()
    for _ in range(config["num_optim_epoch"]):
        order = rng.permutation(len(records))
        for offset in range(0, len(records), batch_size):
            ids = order[offset:offset + batch_size]
            batch = [records[int(i)] for i in ids]
            states = [r["state"] for r in batch]
            target = torch.as_tensor(returns[ids], dtype=torch.float32, device=device)
            adv = torch.as_tensor(advantages[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            if critic_only:
                value = model(states, critic_only=True)
                critic_loss = (value - target).square().mean()
                actor_loss = value.new_zeros(())
                entropy = value.new_zeros(())
                kl = value.new_zeros(())
                clipped = value.new_zeros(())
                loss = config["value_pred_coef"] * critic_loss
            else:
                distribution, value = model(states)
                actions = torch.tensor([r["action"] for r in batch], device=device)
                old = torch.tensor([r["old_log_prob"] for r in batch], device=device)
                log_prob = distribution.log_prob(actions)
                diff = log_prob - old
                if not torch.isfinite(diff).all():
                    raise FloatingPointError("Nonfinite PPO probability ratio.")
                ratio = torch.exp(diff.clamp(-10., 10.) if tail else diff)
                epsilon = config["clip_epsilon"]
                actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1. - epsilon, 1. + epsilon) * adv).mean()
                critic_loss = (value - target).square().mean()
                entropy = distribution.entropy().mean()
                kl = ((ratio - 1.) - diff).mean()
                clipped = ((ratio - 1.).abs() > epsilon).float().mean()
                if tail and not bool(adv.ne(0).any()):
                    value = model(states, critic_only=True)
                    critic_loss = (value - target).square().mean()
                    loss = config["value_pred_coef"] * critic_loss
                else:
                    loss = actor_loss + config["value_pred_coef"] * critic_loss
                    if not tail:
                        loss = loss - config["entropy_coef"] * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss.")
            loss.backward()
            squared_norm = 0.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite model gradient.")
                    squared_norm += float(parameter.grad.detach().square().sum())
            optimizer.step()
            metrics.append([float(actor_loss.detach()), float(critic_loss.detach()),
                            float(entropy.detach()), float(kl.detach()), float(clipped.detach()),
                            math.sqrt(squared_norm)])
    average = np.mean(metrics, axis=0)
    return {"actor_loss": average[0], "critic_loss": average[1], "entropy": average[2],
            "approx_kl": average[3], "clip_fraction": average[4], "grad_norm": average[5],
            "optimizer_steps": len(metrics), "peak_promotion": promote, "critic_only": critic_only,
            "rollout_peak": None if peak is None else peak["plan"]["utility"]}


def checkpoint(model, settings, config, weights, step, mode, seed, selection):
    return {"model": model.state_dict(), "settings": settings, "config": config,
            "weights": weights.tolist(), "step": step, "decoder_mode": mode, "seed": seed,
            "profile": model.profile, "selection": selection,
            "data_fingerprint": load_bundle(settings["city"], settings["initial_lines"])["fingerprint"],
            "code_identity": code_identity()}


def test(model, settings, weights, seeds, mode, output):
    plans, batches = [], []
    with preserve_randomness():
        for index, group in enumerate(seeds):
            batch = [decode(model, settings, weights, group[0], True, mode)]
            batch += [decode(model, settings, weights, seed, False, mode) for seed in group[1:]]
            write_json(output / f"test_batch_{index}.json", batch)
            plans.extend(batch)
            batches.append(max(batch, key=lambda plan: plan["utility"])["utility"])
    best = max(plans, key=lambda plan: plan["utility"])
    write_json(output / "best_plan.json", best)
    result = {"best_of_batch": batches, "mean_best_of_batch": float(np.mean(batches)),
              "best_utility": best["utility"], "best_raw_GJF": best["raw_GJF"],
              "generated_plans": len(plans), "weights": weights.tolist()}
    write_json(output / "evaluation.json", result)
    return result


def prepare_output(path):
    output = Path(path).expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Choose an output directory outside the source project.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing runs are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(args, config):
    target = normalize_weights(args.weights)
    settings = {"city": args.city, "initial_lines": args.initial_lines,
        "budget": args.budget, "max_lines": args.max_lines, "min_stations": args.min_stations}
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    config = copy.deepcopy(config)
    if config["gamma"] != 1.:
        raise ValueError("Undiscounted physical gain requires gamma equal to one.")
    for key in ("mini_batch_size", "num_optim_epoch"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.rollout_steps % config["mini_batch_size"] or args.iterations < 2:
        raise ValueError("Use at least two rollouts and an exact multiple of the minibatch size.")
    env = make_env(settings, target)
    model = ActorCritic(env, config, target).to(device)
    roots = np.random.SeedSequence(args.seed).spawn(3)
    val_seeds = roots[0].generate_state(9).astype(np.uint64).tolist()
    test_seeds = [child.generate_state(17).astype(np.uint64).tolist() for child in roots[1].spawn(3)]
    output = prepare_output(args.output)
    started = time.perf_counter()
    if args.mode == "evaluate":
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if (saved["settings"] != settings or saved["data_fingerprint"] != env.bundle["fingerprint"]
                or saved["config"] != config or saved["code_identity"] != code_identity()):
            raise ValueError("Checkpoint configuration, code or data differs.")
        np.testing.assert_allclose(saved["weights"], target, rtol=0, atol=1e-12)
        model.load_state_dict(saved["model"])
        result = test(model, settings, target, test_seeds, saved["decoder_mode"], output)
        result["seconds"] = time.perf_counter() - started
        print(json.dumps(result), flush=True)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"],
                                  weight_decay=config.get("weightdecay", 0.))
    rng = np.random.default_rng(roots[2])
    current_weights = scheduled_weights(target, model.profile, 0, args.iterations, args.curriculum_power)
    state = env.reset(current_weights, 0)
    episode_id, partial_return = 0, 0.
    history, best_score, total_updates = [], -math.inf, 0
    previous = None
    mode = 0

    def select(step, evaluation):
        nonlocal best_score
        evaluation["step"] = step
        selected = evaluation["selection"] > best_score
        evaluation["selected"] = selected
        write_json(output / f"validation_{step}.json", evaluation)
        if selected:
            best_score = evaluation["selection"]
            torch.save(checkpoint(model, settings, config, target, step, evaluation["decoder_mode"],
                                  args.seed, best_score), output / "best.pt")
        return selected

    previous = validation(model, settings, target, val_seeds, mode)
    select(0, previous)
    for rollout in range(args.iterations):
        current_weights = scheduled_weights(target, model.profile, rollout, args.iterations, args.curriculum_power)
        tail = mode == 1
        snapshot = copy.deepcopy(model.state_dict()) if tail else None
        incumbent = previous["deterministic"]
        for group in optimizer.param_groups:
            group["lr"] = config["lr"] / 4. if tail else config["lr"]
        records, completed, state, episode_id, partial_return = collect(
            model, env, state, args.rollout_steps, current_weights, mode, episode_id, partial_return)
        stats = update(model, optimizer, records, config, rng, tail, incumbent, completed)
        total_updates += stats["optimizer_steps"]
        step = (rollout + 1) * args.rollout_steps
        next_mode = int(model.profile != "commute" and rollout + 1 >= args.iterations - 1)
        evaluation = validation(model, settings, target, val_seeds, next_mode)
        if tail and evaluation["deterministic"] + 1e-8 < incumbent:
            model.load_state_dict(snapshot)
            evaluation = copy.deepcopy(previous)
            stats["policy_rollback"] = True
        else:
            stats["policy_rollback"] = False
        select(step, evaluation)
        prior = None if model.prior_strength is None else float(model.prior_strength.detach())
        entry = {"step": step, "train_weights": current_weights.tolist(), "decoder_mode": mode,
                 "prior_strength": prior, "validation_selection": evaluation["selection"],
                 "validation_deterministic": evaluation["deterministic"],
                 "validation_best": evaluation["best"], "completed_episodes": len(completed),
                 "elapsed_seconds": time.perf_counter() - started, **stats}
        history.append(entry)
        write_json(output / "training.json", history)
        print(json.dumps(entry), flush=True)
        previous, mode = evaluation, next_mode
    saved = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    result = test(model, settings, target, test_seeds, saved["decoder_mode"], output)
    result.update({"environment_steps": args.iterations * args.rollout_steps,
        "optimizer_steps": total_updates, "validation_calls": args.iterations + 1,
        "validation_plans": len(val_seeds) * (args.iterations + 1),
        "best_checkpoint_step": saved["step"], "profile": model.profile,
        "device": str(device), "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU", "seconds": time.perf_counter() - started,
        "data_fingerprint": env.bundle["fingerprint"], "code_identity": code_identity(),
        "training_weight_rule": "power-interpolation" if model.profile == "commute" else "fixed"})
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)


def self_check(args, config):
    """No training, file writes, external assets or persistent caches."""
    from .features import DynamicPotentials
    target = normalize_weights(args.weights)
    settings = {"city": args.city, "initial_lines": args.initial_lines,
        "budget": args.budget, "max_lines": args.max_lines, "min_stations": args.min_stations}
    env = make_env(settings, target)
    initial = env.reset()
    assert initial[1].shape[1] == 40 and len(initial) == 6
    np.testing.assert_allclose(env.raw, 0., atol=1e-10)
    _, vector, done, _ = env.step(env.stop_action)
    assert done and env.plan()["cost"] == 0.
    np.testing.assert_array_equal(vector, np.zeros(3))
    env.reset()
    for _ in range(64):
        allowed = np.flatnonzero(env.observation()[-2][:-1])
        if len(allowed) == 0 or (len(env.inner.actions) >= env.min_stations and not env.core.incomplete):
            env.step(env.stop_action)
            break
        env.step(int(allowed[0]))
        if env.done:
            break
    if not env.done:
        raise AssertionError("Correctness episode exceeded the diagnostic step cap.")
    env.plan()
    np.testing.assert_allclose(env.return_vector, env.raw / env.refs, atol=1e-8)
    a, returns = gae([1., 2., 3.], [0., 0., 0.], [0., 0., 0.], [False, False, True], 1.)
    np.testing.assert_allclose(returns, [6., 5., 3.])
    env.reset()
    model = ActorCritic(env, config, target).cpu()
    model.eval()
    state = env.observation()
    dist, value = model([state])
    action = dist.logits.argmax(-1)
    log_before = dist.log_prob(action).detach()
    dist_again, _ = model([state.clone()])
    torch.testing.assert_close(log_before, dist_again.log_prob(action), atol=1e-6, rtol=0)
    loss = -dist.log_prob(action).mean() + value.square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    sparse_state = state.clone()
    sparse_state.decoder_mode = 1
    sparse_dist, sparse_value = model([sparse_state])
    sparse_action = sparse_dist.logits.argmax(-1)
    assert int(sparse_action.item()) == int(action.item())
    (-sparse_dist.log_prob(sparse_action).mean() + sparse_value.square().mean()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    p = model.potentials
    nodes = torch.tensor(state[1])[None]
    built = (nodes[:, :, 30] > .5).float()
    acc = p.accessibility(built)
    np.testing.assert_allclose(acc.numpy()[0], np.exp(-env.scenario.initial_nearest / 3.), atol=1e-6)
    gain = p.fair_gain(acc)[0].numpy() * 100.
    ids = np.linspace(0, len(env.scenario.nodes) - 1, 9, dtype=int)
    for index in ids:
        nearest = np.minimum(env.scenario.initial_nearest, env.scenario.direct[:, index])
        exact = 100. * np.dot(env.bundle["arrays"]["v"],
            np.exp(-nearest / 3.) - np.exp(-env.scenario.initial_nearest / 3.))
        np.testing.assert_allclose(gain[index], exact, atol=1e-5)
    for count in (0, 1, 5):
        for new_count in (5, 15, 28):
            raw = torch.zeros(1, 7, requires_grad=True)
            mask = torch.zeros_like(raw)
            mask[:, :count] = 1.
            mask[:, -1] = 1.
            potential = legal_zscore(raw[:, :-1], mask)
            target_logits = sparse_targets(raw.detach(), mask, potential.detach(),
                                           torch.tensor([new_count]))
            logits = attach_legal_gradient(raw, target_logits, mask)
            expected = raw.masked_fill(~mask.bool(), -1e9).argmax(-1)
            torch.testing.assert_close(logits.argmax(-1), expected)
            distribution = torch.distributions.Categorical(logits=logits)
            (-distribution.log_prob(expected).mean()).backward()
            assert torch.isfinite(raw.grad).all()
    print(json.dumps({"check": "passed", "nodes": len(env.scenario.nodes),
                      "profile": model.profile, "data_fingerprint": env.bundle["fingerprint"]}), flush=True)
