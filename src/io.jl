using FileIO
using JLD2

export saveGPSLCObject, loadGPSLCObject

function saveGPSLCObject(g::GPSLCObject, filename::String)
    jldopen(filename * ".jld2", "w") do file
        group = JLD2.Group(file, "GPSLCObject")
        for field in fieldnames(GPSLCObject)
            group[string(field)] = getfield(g, field)
        end
    end
end

function loadGPSLCObject(filename::String)
    jldopen(filename * ".jld2", "r") do file
        g = (file["GPSLCObject/"*string(field)] for field in fieldnames(GPSLCObject))
        return GPSLCObject(g...)
    end
end
