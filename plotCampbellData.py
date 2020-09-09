import numpy as np
import pandas as pd
import os

def plotCampbellData(OP, Freq, Damp, sx='WS_[m/s]', UnMapped=None, fig=None, axes=None, ylim=None):
    """ Plot Campbell data as returned by postproMBC """
    import matplotlib.pyplot as plt
    FullLineStyles = [':', '-', '-+', '-o', '-^', '-s', '--x', '--d', '-.', '-v', '-+', ':o', ':^', ':s', ':x', ':d', ':.', '--','--+','--o','--^','--s','--x','--d','--.'];
    Markers    = ['', '+', 'o', '^', 's', 'd', 'x', '.']
    LineStyles = ['-', ':', '-.', '--'];
    Colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    MW_Light_Blue    = np.array([114,147,203])/255.
    MW_Light_Orange  = np.array([225,151,76])/255.
    MW_Light_Green   = np.array([132,186,91])/255.
    MW_Light_Red     = np.array([211,94,96])/255.
    MW_Light_Gray    = np.array([128,133,133])/255.
    MW_Light_Purple  = np.array([144,103,167])/255.
    MW_Light_DarkRed = np.array([171,104,87])/255.
    MW_Light_Kaki    = np.array([204,194,16])/255.
    MW_Blue     =     np.array([57,106,177])/255.
    MW_Orange   =     np.array([218,124,48])/255.
    MW_Green    =     np.array([62,150,81])/255.
    MW_Red      =     np.array([204,37,41])/255.
    MW_Gray     =     np.array([83,81,84])/255.
    MW_Purple   =     np.array([107,76,154])/255.
    MW_DarkRed  =     np.array([146,36,40])/255.
    MW_Kaki     =     np.array([148,139,61])/255.

    def modeStyle(i, lbl):
        lbl=lbl.lower().replace('_',' ')
        lbl=lbl.replace(')','')
        lbl=lbl.replace('(',' ')
        ms = 4
        c  = Colors[np.mod(i,len(Colors))]
        ls = LineStyles[np.mod(int(i/len(Markers)),len(LineStyles))]
        mk = Markers[np.mod(i,len(Markers))]
        # Color
        if any([s in lbl for s in ['1st tower fa', '2nd tower fa']]):
            c=MW_Red;mk='s';ms=6
        elif any([s in lbl for s in ['1st tower ss', '2nd tower ss']]):
            c=MW_Blue;mk='d';ms=6
        elif any([s in lbl for s in ['blade edge  regressive']]):
            c='y'; mk='o'; ms=6;
        elif any([s in lbl for s in ['blade edge  progressive']]):
            c='c'; mk='+'; ms=6;
        elif any([s in lbl for s in ['regressive']]):
            c=MW_Green; mk='v'; ms=6;
        elif any([s in lbl for s in ['collective']]):
            c='brown'; mk='*'; ms=6;
        elif any([s in lbl for s in ['progressive']]):
            c='m'; mk='^'; ms=6;

        # Line style
        # if any([s in lbl for s in ['tower fa','collective','drivetrain']]):
        #     ls='-'
        if any([s in lbl for s in ['2nd']]):
            ls='--'
        else:
            ls='-'
            
        # Marker
        # if any([s in lbl for s in ['fa']]):
        #     mk='s'; ms=4
        # if any([s in lbl for s in ['ss']]):
        #     mk='d'; ms=4
        # elif any([s in lbl for s in ['regressive']]):
        #     mk='v'; ms=6;
        # elif any([s in lbl for s in ['collective']]):
        #     mk='*'; ms=6;
        # elif any([s in lbl for s in ['progressive']]):
        #     mk='^'; ms=6;
        # elif any([s in lbl for s in ['(collective)']]):
        #     mk='^'; ms=8
        # elif any([s in lbl for s in ['(progressive)']]):
        #     mk='^'; ms=8
        # elif any([s in lbl for s in ['collective']]):
        #     mk='*'; ms=8
        # elif any([s in lbl for s in ['blade','tower','drivetrain']]):
        #     mk=''; 
        return c, ls, ms, mk

    # Init figure
    if fig is None:
        fig,axes_ = plt.subplots(1,2)
        fig.set_size_inches(13,7.0,forward=True) # default is (6.4,4.8)
        fig.subplots_adjust(top=0.78,bottom=0.11,left=0.04,right=0.98,hspace=0.06,wspace=0.16)
    if axes is None:
        axes=axes_

    # Estimating figure range
    FreqRange = [0                         , np.nanmax(Freq.iloc[:,:])*1.01]
    DampRange = [np.nanmin(Damp.iloc[:,2:]), np.nanmax(Damp.iloc[:,:])*1.01]
    if ylim is not None:
        FreqRange=ylim
    if DampRange[0]>0:
        DampRange[0]=0

    # Plot "background"

    #exit()
    # Plot mapped modes
    iModeValid=0
    xPlot=[]; yPlot=[]
    for iMode,lbl in enumerate(Freq.columns.values):
        if lbl.find('not_shown')>0:
            # TODO ADD TO UNMAPPED
            continue
        iModeValid+=1
        c, ls, ms, mk = modeStyle(iModeValid, lbl)
        axes[0].plot(OP, Freq[lbl].values, ls, marker=mk, label=lbl.replace('_',' '), markersize=ms, color=c)
        axes[1].plot(OP, Damp[lbl].values, ls, marker=mk                            , markersize=ms, color=c)
        xPlot=np.concatenate((xPlot, OP))
        yPlot=np.concatenate((yPlot, Freq[lbl].values))

    # Unmapped modes (NOTE: plotted after to over-plot)
    if UnMapped is not None:
        axes[0].plot(UnMapped[sx].values, UnMapped['Freq_[Hz]'  ].values, '.', markersize=6, color=[0.5,0.5,0.5])
        axes[1].plot(UnMapped[sx].values, UnMapped['Damping_[-]'].values, '.', markersize=1, color=[0.5,0.5,0.5])
    # Highligh duplicates (also after)
    Points=[(x,y) for x,y in zip(xPlot,yPlot)]
    Plot = pd.Series(Points,dtype=float)
    for xDupl,yDupl in Plot[Plot.duplicated()]:
        axes[0].plot(xDupl,yDupl, 'o',color='r')

    axes[0].set_xlabel(sx.replace('_',' '))
    axes[1].set_xlabel(sx.replace('_',' '))
    axes[0].set_ylabel('Frequencies [Hz]')
    axes[1].set_ylabel('Damping ratios [-]')
    axes[0].legend(bbox_to_anchor=(0., 1.02, 2.16, .802), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
    axes[0].set_ylim(FreqRange)
    
    XLIM=axes[1].get_xlim()
    axes[1].plot(XLIM, [0,0],'-', color='k', lw=0.5)
    axes[1].set_xlim(XLIM)
    axes[1].set_ylim(DampRange)
    path_to_save_fig=os.getcwd()+'/CampbellDiagram.png'
    print('Saving image to ', path_to_save_fig)
    fig.savefig(path_to_save_fig, dpi=300)
    return fig, axes
