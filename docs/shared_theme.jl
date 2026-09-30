# Generated from the owner's catalogue identity through the supported extension hook.
struct SharedHome <: Documenter.Plugin end
function DocumenterVitepress.vitepress_config_transform(::SharedHome, config::String)
    marker = "const nav = ["
    occursin(marker, config) || error("VitePress navigation template changed; review the shared-home adapter")
    return replace(config, marker => marker * "\n  { text: '← cx-xd.org', link: 'https://cx-xd.org/', target: '_self' },"; count=1)
end
