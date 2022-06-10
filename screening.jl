using Distributed

if length(ARGS) != 1
    println("Usage: julia --project screening.jl nprocs")
    exit()
end
nprocs = parse(Int, ARGS[1])
addprocs(nprocs, exeflags="--project")
@everywhere begin

using MKL
using PowerModels
using Ipopt
using ExaAdmm
using LazyArtifacts

abstract type Contingency end
struct LineContingency <: Contingency end
struct GenContingency <: Contingency end

function solve(network, id)
    try
        # if id != 133
        #     result = run_ac_opf(network, optimizer_with_attributes(
        #         Ipopt.Optimizer,
        #         "print_level" => 0, "max_iter" => 200
        #         )
        #     )
        # else
            result = solve_ac_opf(network, optimizer_with_attributes(
                Ipopt.Optimizer,
                "print_level" => 0, "max_iter" => 200
                )
            )
        # end
        if result["termination_status"] == LOCALLY_SOLVED
            return true
        else
            return false
        end
    catch e
        return false
    end
end

function solve_contingency(id::Int, network, ::LineContingency)
    cnetwork = deepcopy(network)
    println("Solving line contingency $id")
    cnetwork["branch"]["$id"]["br_status"] = 0
    return solve(cnetwork, id)
end

function solve_contingency(id::Int, network, ::GenContingency)
    cnetwork = deepcopy(network)
    println("Solving generator contingency $id")
    cnetwork["gen"]["$id"]["gen_status"] = 0
    return solve(cnetwork, id)
end
end

@everywhere begin
    # case = ARGS[1]
    case = "case_ACTIVSg10k"
    # case = "case118"
    const CASE_PATH = joinpath(artifact"ExaData", "ExaData", "matpower", "$case.m")

    network = PowerModels.parse_file(CASE_PATH; import_all=true)
end
PowerModels.export_matpower("export.m", network)

@everywhere begin
    ngen = length(network["gen"])
    nlines = length(network["branch"])

    results_gens = pmap(x -> solve_contingency(x, network, GenContingency()), 1:ngen)
    results_lines = pmap(x -> solve_contingency(x, network, LineContingency()), 1:nlines)
end

println("Found $(length(findall(results_gens))) feasible generator contingencies of a total of $ngen generators: $(findall(results_gens))")
println("Found $(length(findall(results_lines))) feasible line contingencies of a total of $nlines lines: $(findall(results_lines))")


open("$case.gen", "w") do io
    for res in findall(results_gens)
        write(io,"$res\n")
    end
end
open("$case.lines", "w") do io
    for res in findall(results_lines)
        write(io,"$res\n")
    end
end
