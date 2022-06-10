using MPI
using MKL
using PowerModels
using ExaAdmm
using LazyArtifacts
using Ipopt

abstract type Contingency end
struct LineContingency <: Contingency end
struct GenContingency <: Contingency end

function solve(network, id::Int64)
    try
        result = solve_ac_opf(network, optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 10, "max_iter" => 200, "linear_solver" => "ma27"
            )
        )
        if result["termination_status"] == LOCALLY_SOLVED
            return 1
        else
            return 0
        end
    catch e
        return 0
    end
    return 0
end

function solve_contingency(id::Int64, network, ::LineContingency)
    cnetwork = deepcopy(network)
    cnetwork["branch"]["$id"]["br_status"] = 0
    return solve(cnetwork, id)
end

function solve_contingency(id::Int64, network, ::GenContingency)
    cnetwork = deepcopy(network)
    cnetwork["gen"]["$id"]["gen_status"] = 0
    return solve(cnetwork, id)
end
MPI.Init()

# case = ARGS[1]
case = "case_ACTIVSg10k"
case = "case118"
case = "case9"
const CASE_PATH = joinpath(artifact"ExaData", "ExaData", "matpower", "$case.m")

network = PowerModels.parse_file(CASE_PATH; import_all=true)
PowerModels.export_matpower("export.m", network)

ngen = length(network["gen"])
nlines = length(network["branch"])
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
results_gens = zeros(Int64,ngen)
results_lines = zeros(Int64,nlines)
# Generator contingencies
for i in 1:ngen
    if mod(i, nprocs) == rank
        println("Rank $rank with $i and $nprocs Solving generator contingency $i")
        results_gens[i] = solve_contingency(i, network, GenContingency())
    end
end
MPI.Reduce!(results_gens, MPI.SUM, 0, MPI.COMM_WORLD)
if rank == 0
    @show ngen
    println("Found $(length(findall(x -> x == 1, results_gens))) feasible generator contingencies of a total of $ngen generators: $(findall(x -> x == 1, results_gens))")
    open("$case.gen", "w") do io
        for res in findall(x -> x == 1, results_gens)
            write(io,"$res\n")
        end
    end
end

# Line contingencies
for i in 1:nlines
    if mod(i, nprocs) == rank
        println("Solving line contingency $i")
        results_lines[i] = solve_contingency(i, network, LineContingency())
    end
end
MPI.Reduce!(results_lines, MPI.SUM, 0, MPI.COMM_WORLD)

if rank == 0
    @show nlines
    println("Found $(length(findall(x -> x == 1, results_lines))) feasible line contingencies of a total of $nlines lines: $(findall(x -> x == 1, results_lines))")
    open("$case.lines", "w") do io
        for res in findall(x -> x == 1, results_lines)
            write(io,"$res\n")
        end
    end
end
MPI.Finalize()

