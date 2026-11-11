use crate::tool::ToolRegistration;

pub fn register() -> ToolRegistration {
    ToolRegistration {
        type_id: "path",
        display_name: "Path",
        icon: "lucide:spline",
        description: "Draw an S9 interpolating spline: click to place knots, drag to shape. Knots stay editable on the committed vector object.",
        hotkey_action: "pathTool",
        params: &[],
    }
}
