@enum OptimizationGoal MINIMIZE MAXIMIZE
@enum TerminationStatus PENDING SOLVING OPTIMAL UNBOUNDED INFEASIBLE MAX_ITERATIONS_REACHED
@enum ConstraintSense EQ LE GE
Base.@kwdef mutable struct Constraint
    lhs::Vector{Float64} = Float64[]
    sense::ConstraintSense = EQ
    rhs::Float64 = 0.0
end

Base.@kwdef mutable struct LinearModel
    goal::OptimizationGoal = MINIMIZE
    objective::Vector{Float64} = Float64[]
    constraints::Vector{Constraint} = Constraint[]
end

Base.@kwdef mutable struct SimplexModel
    goal::OptimizationGoal = MINIMIZE
    c::Vector{Float64} = Float64[]
    A::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
    b::Vector{Float64} = Float64[]
    basic_vars::Vector{Int} = Int[]
    non_basic_vars::Vector{Int} = Int[]
    two_phase::Bool = false
    iteration::Int = 0
    termination_status::TerminationStatus = PENDING
    solution::Vector{Float64} = Float64[]
    objective_value::Float64 = 0.0
end
