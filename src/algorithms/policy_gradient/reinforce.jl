export ReinforceLearner

struct ReinforceLearner <: AbstractLearner
    approximator::A
end

function (learner::ReinforceLearner)(env)
    env |>
    get_state |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner.approximator), x) |>
            learner.approximator |>
            vec |>
            send_to_host
end

function RLBase.update!(learner::ReinforceLearner, t::AbstractTrajectory)

end