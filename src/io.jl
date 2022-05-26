using Serialization

export saveGPSLCObject, loadGPSLCObject


"""
    saveGPSLCObject(g, filename)
    saveGPSLCObject(g, "path/to/filename")
    saveGPSLCObject(g, "path/to/filename.gpslc")
This function will save the [`GPSLCObject`](@ref) `g` to the file `<filename>.gpslc`. This [`GPSLCObject`](@ref), including the posterior samples contained within it can be retrieved with the [`loadGPSLCObject`](@ref) function.

Note: The extension `.gpslc` is optional and will be added if it is not included.
"""
function saveGPSLCObject(g::GPSLCObject, filename::String)
    if length(filename) > 6 && filename[end-5:end] == ".gpslc"
        filename = filename[1:end-6]
    end
    serialize(filename * ".gpslc", g)
end

"""
    loadGPSLCObject(filename)
    loadGPSLCObject("path/to/filename")
    loadGPSLCObject("path/to/filename.gpslc")
This function will load and return the [`GPSLCObject`](@ref) contained in `<filename>.gpslc`. 
    
Note: the extension `.gpslc` is optional and will be added if it is not included.
"""
function loadGPSLCObject(filename::String)
    if length(filename) > 6 && filename[end-5:end] == ".gpslc"
        filename = filename[1:end-6]
    end
    deserialize(filename * ".gpslc")
end