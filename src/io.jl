using Serialization

export saveGPSLCObject, loadGPSLCObject


"""
    saveGPSLCObject(g, filename)
This function will save the GPSLCObject `g` to the file `<filename>.gpslc`. This GPSLCObject, including the posterior samples contained within it can be retrieved with the [`loadGPSLCObject`](@ref) function.
"""
function saveGPSLCObject(g::GPSLCObject, filename::String)
    serialize(filename * ".gpslc", g)
end

"""
    loadGPSLCObject(filename)
This function will load and return the GPSLCObject contained in `<filename>.gpslc`.
"""
function loadGPSLCObject(filename::String)
    deserialize(filename * ".gpslc")
end