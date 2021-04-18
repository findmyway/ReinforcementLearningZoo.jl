export QRDQNLearner

mutable struct QRDQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    min_replay_history::Int
    update_freq::Int
    update_step::Int
    target_update_freq::Int
    sampler::NStepBatchSampler
    ensemble_num::Int
    rng::R
    κ::Float32
    τ::Vector{Float64}
    loss::Float32
end

"""
    QRDQNLearner(;kwargs...)

See paper: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `ensemble_num::Int=1`: the number of ensemble approximators.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `traces = SARTS`, set to `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `κ = Float32`, κ in quantile_huber_loss set to 1.0f0.
- `rng = Random.GLOBAL_RNG`
"""
function QRDQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    ensemble_num::Int = 1,
    target_update_freq::Int = 100,
    traces = SARTS,
    update_step = 0,
    κ::Float32 = 1.0f0,
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    N = ensemble_num
    τ = collect(0:N+1)/N
    τ  = (τ[2:N+1]+τ[1:N])/2
    QRDQNLearner(
        approximator,
        target_approximator,
        loss_func,
        min_replay_history,
        update_freq,
        update_step,
        target_update_freq,
        sampler,
        N,
        rng,
        κ,
        τ,
        0.0f0,
    )
end

Flux.functor(x::QRDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::QRDQNLearner)(env)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = reshape(learner.approximator(s), :, learner.ensemble_num)
    vec(mean(q, dims = 2)) |> send_to_host
end

function RLBase.update!(learner::QRDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    loss_func = learner.loss_func
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    N = learner.ensemble_num
    κ = learner.κ
    τ = learner.τ
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(repeat(batch.action, inner = N), 1:(N*batch_size))

    target_q = reshape(Qₜ(s), :, N, batch_size)
    avg_q = mean(target_q, dims=2)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        avg_q .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    aₜ = argmax(avg_q, dims=1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:N-1, 0:0)))
    qₜ = reshape(target_q[aₜ], :, batch_size)
    target =
    reshape(r, 1, batch_size) .+ γ * reshape(1 .- t, 1, batch_size) .* qₜ

    gs = gradient(params(Q)) do
        q = reshape(Q(s), :, N*batch_size)
        q = q[a]

        target = reshape(target, N, 1, batch_size)
        q = reshape(q, 1, N, batch_size)

        TD_error = (target .- q)
        temp = Zygote.dropgrad(abs.(TD_error) .<  κ)
        element_wise_huber_loss = ((TD_error.^2) .* temp) + κ*(TD_error .- 0.5*κ) .* (1 .- temp)

        # dropgrad
        element_wise_huber_loss =
            abs.(reshape(τ, 1, N) .- Zygote.dropgrad(TD_error .< 0)) .*
            element_wise_huber_loss ./ κ
        batch_quantile_huber_loss = mean(sum(element_wise_huber_loss; dims = 1), dims=1)
        quantile_huber_loss =
            mean(batch_quantile_huber_loss)
        ignore() do
            learner.loss = quantile_huber_loss
        end
        quantile_huber_loss
    end

    update!(Q, gs)
end
