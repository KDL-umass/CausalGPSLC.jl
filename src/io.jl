using FileIO
using JLD2

export saveGPSLCObject, loadGPSLCObject

function saveGPSLCObject(g::GPSLCObject, filename::String)
    jldopen(filename * ".jld2", "w") do file
        group = JLD2.Group(file, "GPSLCObject")
        for field in fieldnames(GPSLCObject)
            if field == "posteriorSamples"
                ps = getfield(g, field)
                nps = length(ps)
                for i in 1:nps
                    for (addr, val) in get_values_shallow(ps[i])
                        group[string(field)][i][addr] = val
                    end
                end
            else
                group[string(field)] = getfield(g, field)
            end
        end
        println("file $(file["GPSLCObject"]["posteriorSamples"][1][:Y])")
    end
end


function loadGPSLCObject(filename::String)
    jldopen(filename * ".jld2", "r") do file
        println("file $(file["GPSLCObject"])")
        nps = length(file["GPSLCObject"]["posteriorSamples"])

        posteriorSamples = []
        for i in 1:nps
            p = choicemap([
                (k, file["GPSLCObject"]["posteriorSamples"][i][k])
                for k in keys(file["GPSLCObject"]["posteriorSamples"][i])
            ]...)
            println(p)
            push!(posteriorSamples, p)
        end
        println(posteriorSamples)

        g = (
            (
                field == "posteriorSamples" ?
                posteriorSamples : file["GPSLCObject"][string(field)]
            )
            for field in keys(file["GPSLCObject"])
        )
        return GPSLCObject(g...)
    end
end