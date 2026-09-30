# Use DocumenterVitepress's extension hook so its standard theme, search and
# scientific rendering remain intact. The checked-in CSS is independently loaded
# through the supported docs/src/.vitepress/theme/overrides.css convention.
struct SharedHome <: Documenter.Plugin end
function DocumenterVitepress.vitepress_config_transform(::SharedHome, config::String)
    marker = "const nav = ["
    occursin(marker, config) || error("VitePress navigation template changed; review the shared-home adapter")
    return replace(config, marker => marker * "\n  { text: '← s-am-i.com', link: 'https://s-am-i.com/' },"; count=1)
end
