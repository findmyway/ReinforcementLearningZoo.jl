export DeepCFRPolicy


struct DeepCFRPolicy <: AbstractPolicy
end

function DeepCFRPolicy(;
    env,
    n_iter,
    n_traversal,
    rng = Random.GLOBAL_RNG,
)
    for iter in 1:n_iter  # iter is used for Linear CFR
        for p in get_players(env)
            if p != get_chance_player(env)
                for _ in 1:n_traversal
                    traverse(copy(env), p, rng)
                end
                # TODO: reinitialize advantage network
                # TODO: train advantage network
            end
        end
    end
    # TODO: train strategy network
end

"DeepCFR Traversal with External Sampling"
function traverse(env, p, rng)
    current_player = get_current_player(env)

    if get_terminal(env)
        get_reward(env, p)
    elseif current_player == get_chance_player(env)
        env(rand(rng, get_actions(env)))
        traverse(env, p, rng)
    else
        V = adv_nets[p](get_state(env))
        σ = regret_matching(V)

        if current_player == p
            # TODO: update v memory
        else
            # TODO: update σ memory
            a′ = sample(rng, Weights(σ, 1.0))
            env(get_legal_actions(env)[a′])
            traverse(env, p, rng)
        end
    end
end
