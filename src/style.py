import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

def set_plotly_defaults():
    # Start from a clean built-in template
    base = pio.templates["plotly_white"]
    tpl = go.layout.Template(base)

    # Global “house style”
    tpl.layout.update(
        hovermode="x unified",
        margin=dict(l=60, r=25, t=70, b=55),
        height=520,
        font=dict(size=14),
        title=dict(x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=.98, xanchor="left", x=0.02),
        xaxis=dict(showgrid=True, rangeslider=dict(visible=True)),
        yaxis=dict(showgrid=True, zeroline=False, tickformat=".2f"),
    )

    # Register + set as default
    pio.templates["heri_white"] = tpl
    pio.templates.default = "heri_white"
    px.defaults.template = "heri_white"

    # Optional: defaults for px layout size too (if you want)
    px.defaults.height = 520
    # px.defaults.width = 950  # optional

set_plotly_defaults()
