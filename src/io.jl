using Serialization

export saveGPSLCObject, loadGPSLCObject

function saveGPSLCObject(g::GPSLCObject, filename::String)
    serialize(filename * ".gpslc", g)
end


function loadGPSLCObject(filename::String)
    deserialize(filename * ".gpslc")
end