using Coverage
using LocalCoverage

function simpleCoverage()
    # process '*.cov' files
    coverage = process_folder() # defaults to src/; alternatively, supply the folder name as argument
    # coverage = append!(coverage, process_folder("deps"))  # useful if you want to analyze more than just src/

    # process '*.info' files, if you collected them
    coverage = merge_coverage_counts(coverage, filter!(
        let prefixes = (joinpath(pwd(), "src", "*.jl"))#,joinpath(pwd(), "deps", ""))
            c -> any(p -> startswith(c.filename, p), prefixes)
        end,
        LCOV.readfolder("test"))
    )

    # Get total coverage for all Julia files
    covered_lines, total_lines = get_summary(coverage)

    println("Total percentage: $(covered_lines / total_lines)")

    LCOV.writefile("coverage-lcov.info", coverage)
end

"""
Generate local coverage report with web interface to explore
files that are uncovered.
"""
function interactiveCoverage()
    generate_coverage("GPSLC", genhtml=true, show_summary=true, genxml=false)
    open_coverage("GPSLC")
end

interactiveCoverage()