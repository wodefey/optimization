using LinearAlgebra
include("types.jl")
include("utils.jl")

function optimize(lm::LinearModel; verbose=false, max_iterations=100)
    verify(lm)

    sm = build_simplex_model(lm)
    simplex!(sm; verbose=verbose, max_iterations=max_iterations)

    if sm.two_phase && (sm.termination_status == OPTIMAL)
        sm = build_simplex_model(lm, basic_vars=sm.basic_vars)
        simplex!(sm; verbose=verbose, max_iterations=max_iterations)
    end

    return sm
end

function simplex!(model::SimplexModel; verbose=false, max_iterations=100)
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
        printstyled("Starting simplex method\n", color=:blue, underline=true)
        println("\tInitial solution x: ", model.solution)
        println("\tInitial basic variables: ", model.basic_vars)
        println("\tInitial non-basic variables: ", model.non_basic_vars)
        println("\tInitial objective value c * x: ", model.c ⋅ model.solution)
        println()
    end

    solving = true
    model.iteration = 0
    model.termination_status = SOLVING
    while solving && model.iteration < max_iterations
        # lnbv = leaving nonbasic variable
        lnbv = 0
        lnbv_idx = 0
        optimum_reduced_cost = 0.0
        optimum_path = []

        model.iteration += 1
        if verbose
            println()
            printstyled("Iteration: ", color=:cyan)
            println(model.iteration)
            printstyled("Determine entering basic variable\n", color=:cyan)
        end

        F = lu(model.A[:, model.basic_vars])
        for (nbv_idx, nbv) ∈ enumerate(model.non_basic_vars)
            path = zeros(length(model.c))
            path[nbv] = 1
            path[model.basic_vars] = F \ (-model.A[:, nbv])
            reduced_cost = model.c ⋅ path
            if verbose
                printstyled("Evaluating non-basic variable: ", color=:yellow)
                println(nbv)
                println("\tPath: ", path)
                println("\tImprovement: ", reduced_cost)
            end
            if (model.goal == MAXIMIZE && reduced_cost > optimum_reduced_cost) ||
               (model.goal == MINIMIZE && reduced_cost < optimum_reduced_cost)
                optimum_reduced_cost = reduced_cost
                lnbv = nbv
                lnbv_idx = nbv_idx
                optimum_path = path
            end
        end

        if lnbv == 0
            solving = false
            model.objective_value = round(model.c ⋅ model.solution, digits=2)
            if model.two_phase && (abs(model.objective_value) > 0.0001)
                model.termination_status = INFEASIBLE
                if verbose
                    println()
                    printstyled("Phase 1 solution is infeasible.", color=:red, italic=true)
                end
            else
                model.termination_status = OPTIMAL
                if verbose
                    printstyled("Optimal solution found.", color=:green)
                end
            end
            return
        end

        if verbose
            printstyled("Entering basic variable: ", color=:yellow)
            println(lnbv)
            println("\tOptimum path: ", optimum_path)
            println("\tOptimum improvement: ", optimum_reduced_cost)
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
            printstyled("\tkey in ratios dictionary is index in basic variables array\n", color=:light_green)
            println("\tRatios: ", ratios)
            println("\tStep: ", step)
            printstyled("Leaving basic variable: ", color=:yellow)
            println(lbv)
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
        push!(model.progress, model.objective_value)
        if verbose
            println("\tUpdated solution x: ", model.solution)
            println("\tUpdated basic variables: ", model.basic_vars)
            println("\tUpdated non-basic variables: ", model.non_basic_vars)
            println("\tUpdated objective value c * x: ", model.objective_value)
            println()
        end
    end

    if model.iteration == max_iterations
        model.termination_status = MAX_ITERATIONS_REACHED
        printstyled("Maximum number of iterations reached.", color=:red)
        println()
    end
end