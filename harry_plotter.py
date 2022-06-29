import asyncio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import typing as t
from matplotlib.axes import Axes

class _Subplot:
    
    def __init__(self,
                 name: str,
                 line_names: t.List[str],
                 x_label: str,
                 y_lim: t.Tuple[float, float],
                 ax: Axes,
                 line_color: t.Optional[t.List[str]] = None) -> None:
        self.name = name
        self.line_names = line_names
        self.ax = ax
        self.buffers: t.Dict[str, t.List[t.Union[float, int]]] = {name:[] for name in line_names}
        self.colors = {name:color for name, color in zip(line_names, line_color)} if line_color else None
        self.x_label = x_label 
        self.y_lim = y_lim

    def update(self, feed_dict: t.Dict[str, t.Union[float, int]]):
        self.ax.cla()
        for key in feed_dict:
            self.buffers[key].append(feed_dict[key])
            
        for key, buffer in self.buffers.items():
            self.ax.plot(buffer, color=self.colors[key] if self.colors else None)
            
        self.ax.legend(self.line_names, bbox_to_anchor=(1,1,1,0), loc="upper left")
        
        self.ax.set_xlabel(self.x_label)
        self.ax.set_title(self.name, fontdict={'horizontalalignment':'right', 'fontweight':'normal', 'verticalalignment':'center'})
      
    def close(self): self.ax.clear()  
    
class LearningCurvePlot:
    
    def __init__(self,
        plot_names=["plot"],
        line_names: t.Union[t.List[str],t.Dict[str, t.List[str]]]=["line-1"],
        line_colors: t.Optional[t.Dict[str, t.List[str]]]=None,
        y_lim=[None, None],
        x_label: t.Union[str, t.List[str]] ="iteration",
        loop: asyncio.AbstractEventLoop = None):
        
        self.loop = asyncio.get_event_loop() if not loop else loop
        self.fig, self.axes = plt.subplots(nrows=len(plot_names), ncols=1, figsize=(16, 4*(len(plot_names))), tight_layout=True)
        self.fig.tight_layout(h_pad=5, w_pad=10)
        self.fig.suptitle('Learning curve', fontsize=16)
        
        plt.subplots_adjust(bottom=0.2, top=0.9)
        if not isinstance(self.axes, np.ndarray): self.axes=[self.axes]
        self.subplots = {name:_Subplot(name, 
                                  line_names if not isinstance(line_names, dict) else line_names[name],
                                  x_label[i] if not isinstance(x_label, str) else x_label,
                                  y_lim if not isinstance(y_lim, dict) else y_lim[name],
                                  ax,
                                  line_colors[name] if line_colors else None) for i, name, ax in zip(range(len(plot_names)),
                                                                                                     plot_names,
                                                                                                     self.axes)}
        
    async def __update(self, feed_dict: t.Dict[str, t.Dict[str, t.Union[float, int]]]):
        
        for key in feed_dict:
            self.subplots[key].update(feed_dict[key])
            
        self.fig.canvas.draw()
        await asyncio.sleep(0.1)
        
        
    def update(self, feed_dict: t.Dict[str, t.Dict[str, t.Union[float, int]]]):
        self.loop.create_task(self.__update(feed_dict))
        
    def close(self):
        [s.close() for s in self.subplots.values()]
        self.fig.clear()
        plt.close(self.fig)