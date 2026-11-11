add_goal!(model::LinearModel, goal::OptimizationGoal) = (model.goal = goal)
add_objective!(model::LinearModel, objective::Vector{Float64}) = (model.objective = objective)
add_constraint!(model::LinearModel, constraint::Constraint) = push!(model.constraints, constraint)

# In this version, we assume:
#   the objective and constraints are zero filled where variables are missing
#   variables are aligned in the same order across the objective and constraints
#   con.sense == EQ NOT CONSIDERED IN SLACK/ARTIFICIAL VARIABLE ADDITION
function build_simplex_model(lm::LinearModel; basic_vars=Int[])
    sm = SimplexModel()
    sm.goal = lm.goal
    sm.c = lm.objective

    m, n = length(lm.constraints), length(lm.objective)
    sm.A = zeros(m, n)
    sm.b = zeros(m)

     for (i, con) in enumerate(lm.constraints)
        sm.A[i, :] = con.lhs
        sm.b[i] = con.rhs
     end

    sm.two_phase = false
    sm.basic_vars = []

    for (i, con) in enumerate(lm.constraints)
         if con.sense == LE
            col = zeros(m,1)
            col[i] = 1
            sm.A = hcat(sm.A,col)
            sm.c = vcat(sm.c, 0.0)
            n += 1
            push!(sm.basic_vars, n)
        elseif con.sense == GE
            sm.two_phase = true
            col = zeros(m,1)
            col[i] = -1
            sm.A = hcat(sm.A,col)
            sm.c = vcat(sm.c, 0.0)
            n += 1
        end
    end

    if isempty(basic_vars)
        if sm.two_phase
            sm.goal = MINIMIZE
            sm.c = zeros(length(sm.c))
            for (i, con) in enumerate(lm.constraints)
                if con.sense == GE
                    col = zeros(m,1)
                    col[i] = 1
                    sm.A = hcat(sm.A,col)
                    sm.c = vcat(sm.c, 1.0)
                    n += 1
                    push!(sm.basic_vars, n)
                end
            end
        end
    else
        sm.basic_vars = basic_vars
        sm.two_phase = false
    end

    sm.non_basic_vars = setdiff(1:n, sm.basic_vars)

    return sm
end

function restart!(model::SimplexModel)
    model.iteration = 0
    model.termination_status = PENDING
    model.solution = zeros(length(model.c))
    model.objective_value = 0.0
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

