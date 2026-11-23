add_goal!(model::LinearModel, goal::OptimizationGoal) = (model.goal = goal)
add_objective!(model::LinearModel, objective::Vector{Float64}) = (model.objective = objective)
add_constraint!(model::LinearModel, constraint::Constraint) = push!(model.constraints, constraint)

struct LinearModelError <: Exception
    msg::String
end

Base.showerror(io::IO, e::LinearModelError) = printstyled(io, "LinearModelError: $(e.msg)\n", color=:red)

function verify(lm::LinearModel)

    if isempty(lm.constraints)
        throw(LinearModelError("No constraints specified in the linear model."))
    end

    if isempty(lm.objective)
        throw(LinearModelError("No objective specified in the linear model."))
    end
end

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

     for (k, con) in enumerate(lm.constraints)
        sm.A[k, :] = con.lhs
        sm.b[k] = con.rhs
     end

    sm.two_phase = false
    sm.basic_vars = []
    sm.slack_vars = []
    sm.artificial_vars = []
     # add slack variables
    for (k, con) in enumerate(lm.constraints)
         if con.sense == LE
            col = zeros(Float64, m, 1)
            col[k] = 1.0
            sm.A = hcat(sm.A,col)
            sm.c = vcat(sm.c, 0.0)
            n += 1
            push!(sm.slack_vars, n)
        elseif con.sense == GE
            # sm.two_phase = true
            col = zeros(Float64, m, 1)
            col[k] = -1.0
            sm.A = hcat(sm.A,col)
            sm.c = vcat(sm.c, 0.0)
            n += 1
            push!(sm.slack_vars, n)
        end
    end

    if isempty(basic_vars)
        # add artificial variables
        for (k, con) in enumerate(lm.constraints)
            add_artificial = false
            row = sm.A[k,:]
            slack_var = filter(x -> row[x] != 0.0, sm.slack_vars)
             if isempty(slack_var)
                # equality constraint
                add_artificial = true
            else
                b_sign = sign(sm.b[k])
                col = slack_var[1]
                col_sign = sign(sm.A[k, col])
                if b_sign != col_sign
                    add_artificial = true
               else
                    push!(sm.basic_vars, col)
                end
            end
            if add_artificial
                col = zeros(Float64, m, 1)
                col[k] = 1.0
                sm.A = hcat(sm.A,col)
                sm.c = vcat(sm.c, 1.0)
                n += 1
                push!(sm.artificial_vars, n)
                push!(sm.basic_vars, n)
            end
        end
        if !isempty(sm.artificial_vars)
            sm.two_phase = true
            sm.goal = MINIMIZE
            sm.c = zeros(length(sm.c))
            sm.c[sm.artificial_vars] .= 1.0
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

