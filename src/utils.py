import base64, os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def save_dashboard(charts: dict, title: str, output_path: str, kpis: dict = None):
    """Save a dict of {chart_title: matplotlib_figure} as a self-contained HTML file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    kpi_html = ""
    if kpis:
        kpi_html = '<div class="kpi-grid">'
        for k, v in kpis.items():
            kpi_html += f'<div class="kpi"><div class="kpi-val">{v}</div><div class="kpi-label">{k}</div></div>'
        kpi_html += '</div>'

    charts_html = ""
    for name, fig in charts.items():
        b64 = fig_to_base64(fig)
        charts_html += f'<div class="chart-card"><h3>{name}</h3><img src="data:image/png;base64,{b64}"/></div>\n'
        plt.close(fig)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#222;padding:24px}}
h1{{font-size:1.8rem;color:#1a3a5c;border-left:5px solid #2563eb;padding-left:12px;margin-bottom:20px}}
.kpi-grid{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px}}
.kpi{{background:#2563eb;color:#fff;border-radius:10px;padding:18px 24px;min-width:140px;text-align:center;box-shadow:0 2px 8px rgba(37,99,235,.3)}}
.kpi-val{{font-size:1.6rem;font-weight:700}}
.kpi-label{{font-size:.8rem;opacity:.85;margin-top:4px}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}}
.chart-card{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,.08)}}
.chart-card h3{{font-size:1rem;color:#1a3a5c;margin-bottom:10px;font-weight:600}}
.chart-card img{{width:100%;border-radius:4px}}
footer{{margin-top:32px;color:#888;font-size:.8rem;text-align:center}}
</style></head><body>
<h1>{title}</h1>{kpi_html}
<div class="chart-grid">{charts_html}</div>
<footer>Jay Desai | jayd409@gmail.com</footer>
</body></html>"""
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Dashboard saved → {output_path}")
