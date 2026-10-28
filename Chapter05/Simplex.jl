using LinearAlgebra

@enum OptimizationGoal MINIMIZE MAXIMIZE
@enum TerminationStatus PENDING SOLVING OPTIMAL UNBOUNDED INFEASIBLE MAX_ITERATIONS_REACHED

Base.@kwdef mutable struct SimplexModel
    goal::OptimizationGoal = MINIMIZE
    c::Vector{Float64} = Float64[]
    A::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
    b::Vector{Float64} = Float64[]
    basic_vars::Vector{Int} = Int[]
    non_basic_vars::Vector{Int} = Int[]
    iteration::Int = 0
    termination_status::TerminationStatus = PENDING
    solution::Vector{Float64} = Float64[]
    objective_value::Float64 = 0.0
end

function Base.show(io::IO, ::MIME"text/plain", m::SimplexModel)
    printstyled(io, "Model:\n", color=:blue, bold=true)
    for field in fieldnames(SimplexModel)
        val = getfield(m, field)
        
        print(io, "  ")
        printstyled(io, lpad(string(field), 18), color=:yellow, bold=true)
        println(io, " : ", val)
    end
end

function simplex(model::SimplexModel; verbose=true, max_iterations=100)
    A_b = model.A[:, model.basic_vars]
    x_b = A_b \ model.b

    model.solution = zeros(length(model.c))
    model.solution[model.basic_vars] = x_b

    if any(x -> x < 0.0, model.solution)
        model.termination_status = INFEASIBLE
        if verbose
            println("Initial solution is infeasible: ", model.solution)
        end
        return
    end

    if verbose
        println("Initial solution x: ", model.solution)
        println("Initial basic variables: ", model.basic_vars)
        println("Initial non-basic variables: ", model.non_basic_vars)
        println("Initial objective value c * x: ", model.c ⋅ model.solution)
    end

    solving = true
    model.iteration = 0
    model.termination_status = SOLVING
    while solving && model.iteration < max_iterations
        # lnbv = leaving nonbasic variable
        lnbv = 0
        lnbv_idx = 0
        # optimum_improvement = model.goal == MINIMIZE ? 1 : -1
        optimum_improvement = 0.0
        optimum_path = []

        for (nbv_idx, nbv) ∈ enumerate(model.non_basic_vars)
            path = zeros(length(model.c))
            path[nbv] = 1
            path[model.basic_vars] = model.A[:, model.basic_vars] \ (-model.A[:, nbv])
            path_improvement = model.c ⋅ path
            if verbose
                println("Path: ", path)
                println("Improvement: ", path_improvement)
            end
            if (model.goal == MAXIMIZE && path_improvement > optimum_improvement) ||
               (model.goal == MINIMIZE && path_improvement < optimum_improvement)
                optimum_improvement = path_improvement
                lnbv = nbv
                lnbv_idx = nbv_idx
                optimum_path = path
            end
        end

        if lnbv == 0
            solving = false
            model.termination_status = OPTIMAL
            model.objective_value = round(model.c ⋅ model.solution, digits=2)
            break   
        end

        model.iteration += 1

        if verbose
            println("Iteration: ", model.iteration)

            println("Leaving nonbasic variable: ", lnbv)
            println("Optimum path: ", optimum_path)
            println("Optimum improvement: ", optimum_improvement)
        end

        # Determine largest step along the best path
        ratios = Dict{Int, Float64}()
        for (bv_idx, bv) ∈ enumerate(model.basic_vars)
            if optimum_path[bv] < 0
                ratios[bv_idx] = -model.solution[bv] / optimum_path[bv]
            end
        end

        if isempty(ratios)
            solving = false
            model.termination_status = UNBOUNDED
            break
        end
        step, lbv_idx = findmin(ratios)
        lbv = model.basic_vars[lbv_idx]
        if verbose
            println("Ratios: ", ratios)
            println("Step: ", step)
            println("Leaving basic variable: ", lbv)
        end

        # Determine the new solution after taking the step along the best path
        x1 = model.solution + step * optimum_path

        # Update the basic and non-basic variables after the pivot
        model.basic_vars[lbv_idx] = lnbv
        sort!(model.basic_vars)
        model.non_basic_vars[lnbv_idx] = lbv
        sort!(model.non_basic_vars)
        model.solution = x1
        model.objective_value = round(model.c ⋅ model.solution, digits=2)
        if verbose
            println("Updated solution x: ", model.solution)
            println("Updated basic variables: ", model.basic_vars)
            println("Updated non-basic variables: ", model.non_basic_vars)
            println("Updated objective value c * x: ", model.objective_value)
            println()
        end
    end

    if model.iteration == max_iterations
        model.termination_status = MAX_ITERATIONS_REACHED
        printstyled("Maximum number of iterations reached.", color=:red)
        println()
    end
end