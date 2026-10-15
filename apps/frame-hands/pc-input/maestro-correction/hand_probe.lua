local function v(t) return string.format("%.3f %.3f %.3f", t.X, t.Y, t.Z) end
local function r(t) return string.format("%.3f %.3f %.3f", t.Pitch, t.Yaw, t.Roll) end
local function q(t) return string.format("%.5f %.5f %.5f %.5f", t.X, t.Y, t.Z, t.W) end
local cams = FindAllOf("CameraComponent") or {}
out("cameras: " .. #cams)
for i, c in ipairs(cams) do if not c:GetOuter():GetFullName():find("MenuPawn") then goto continue end local ok, e = pcall(function() out(string.format("CAM %d %s loc=%s quat=%s", i, c:GetFullName(), v(c:K2_GetComponentLocation()), r(c:K2_GetComponentRotation()))) end); if not ok then out("cam err " .. tostring(e)) end ::continue:: end
for _, sm in ipairs(FindAllOf("SkeletalMeshComponent") or {}) do
    local ok, e = pcall(function()
        local mesh = sm.SkeletalMesh; local name = mesh and mesh:GetFullName() or ""
        if not name:find("Conductor_hand") then return end
        out(string.format("GLOVE %s loc=%s quat=%s", sm:GetFullName(), v(sm:K2_GetComponentLocation()), r(sm:K2_GetComponentRotation())))
        local ok2, e2 = pcall(function() local parent = sm:GetAttachParent(); if parent and parent:IsValid() then out(string.format("  PARENT %s loc=%s quat=%s", parent:GetFullName(), v(parent:K2_GetComponentLocation()), r(parent:K2_GetComponentRotation()))) else out("  PARENT none") end end)
        if not ok2 then out("  parent err " .. tostring(e2)) end
        local n = sm:GetNumBones(); out("  bones " .. n)
        for b = 0, n - 1 do
            local ok3, e3 = pcall(function()
                local bn = sm:GetBoneName(b)
                local st = sm:GetSocketTransform(bn, 0)
                out(string.format("  bone %2d %-14s loc=%s quat=%s", b, bn:ToString(), v(st.Translation), q(st.Rotation)))
            end)
            if not ok3 then out("  bone " .. b .. " err " .. tostring(e3)) end
        end
    end)
    if not ok then out("glove err " .. tostring(e)) end
end
out("probe4 done")