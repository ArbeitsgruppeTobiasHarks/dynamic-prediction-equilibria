
from dataclasses import dataclass
from typing import List


@dataclass
class BoxPlot:
    label: str
    color: str
    data: List[float]


def make_tikz_boxplot(ylabel: str, plots: List[BoxPlot]):
    tikz = """\\begin{tikzpicture}
        \\begin{axis}
  [
  width=.5\\textwidth,
  boxplot/draw direction = y,
  ylabel = {""" + ylabel + """},
  %ymode = log,
  log basis y={10},
  ymajorgrids=true,
  grid style=dashed,
  xtick = {""" + ",".join(str(i) for i in range(1, len(plots) + 1)) + """},
  xticklabels = {""" + ",".join(plot.label for plot in plots) + """},
  every axis plot/.append style = {fill, fill opacity = .1},
  ]
    """
    for plot in plots:
        tikz += """\\addplot + [
          mark = *,
          boxplot,
          solid,
          color="""
        tikz += plot.color
        tikz += """]
          table [row sep = \\\\, y index = 0] {
        data \\\\
        """
        for y in plot.data:
            tikz += f"{y}\\\\\n"
        tikz += "};\n"

    tikz += """\\end{axis}
    \\end{tikzpicture}
    """
    return tikz
