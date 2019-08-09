import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob as glob
from ipywidgets import widgets, interact
from IPython.display import display
from scipy.optimize import curve_fit


def gauss_func(x,w):
    uw = abs(w)
    amp = 1.0/(w*np.sqrt(2.0*np.pi))
    return amp*np.exp(-((x)**2.0)/(2.0*uw**2.0))


def lorz_func(x,w):
    #return 1.0/(x**2+1.0)
    uw = abs(w)
    return (1.0/np.pi)*(.5*uw)/(x**2.0 + (.5*uw)**2.0)


def pv_func(x,w,mixing,amp):
    val_zero = ((lorz_func(0,w))*(1-mixing)) + amp*(gauss_func(0,w)*mixing)
    return (amp/val_zero)*(((lorz_func(x,w))*(1-mixing)) + amp*(gauss_func(x,w)*mixing))


def fit_pv_to_sq(q,sq,qmin=10,qmax=25,first_guess=([30,.5,2])):
    ### initial params are w=30, mixing=.5, amp=2 (unless specific with first_guess=([30,.5,2]))
    qcut, sqcut = cut_data(q,sq,qmin,qmax)
    popt, pcov = curve_fit(pv_func, qcut, sqcut, p0=first_guess,
                           bounds=([1,0,.000001],[30,1,10]))
    return np.array(pv_func(q,*popt))


def pv_v0_func(x,w,mixing,amp,c0):
    val_zero = ((lorz_func(0,w))*(1-mixing)) + amp*(gauss_func(0,w)*mixing)
    return (amp/val_zero)*(((lorz_func(x,w))*(1-mixing)) + amp*(gauss_func(x,w)*mixing)) + c0


def fit_pv_n0_to_sq(q,sq,qmin=10,qmax=25,first_guess=([30,.5,2,0])):
    qcut, sqcut = cut_data(q,sq,qmin,qmax)
    popt, pcov = curve_fit(pv_v0_func, qcut, sqcut, p0=first_guess,
                           bounds=([1,0,.000001,-100],[30,1,10,100]))
    return np.array(pv_v0_func(q,*popt))


def pv_v1_func(x,w,mixing,amp,c0,c1):
    val_zero = ((lorz_func(0,w))*(1-mixing)) + amp*(gauss_func(0,w)*mixing)
    return (amp/val_zero)*(((lorz_func(x,w))*(1-mixing)) + amp*(gauss_func(x,w)*mixing)) + c0 +x*c1


def fit_pv_n1_to_sq(q,sq,qmin=10,qmax=25,first_guess=([30,.5,2,0,0])):
    qcut, sqcut = cut_data(q,sq,qmin,qmax)
    popt, pcov = curve_fit(pv_v1_func, qcut, sqcut, p0=first_guess,
                           bounds=([1,0,.000001,-10,-10],[30,1,10,10,10]))
    return np.array(pv_v1_func(q,*popt))


def pv_v2_func(x,w,mixing,amp,c0,c1,c2):
    val_zero = ((lorz_func(0,w))*(1-mixing)) + amp*(gauss_func(0,w)*mixing)
    return (amp/val_zero)*(((lorz_func(x,w))*(1-mixing)) + amp*(gauss_func(x,w)*mixing)) + c0 +x*c1+x*c2**2


def fit_pv_n2_to_sq(q,sq,qmin=10,qmax=25,first_guess=([30,.5,2,0,0,0])):
    qcut, sqcut = cut_data(q,sq,qmin,qmax)
    popt, pcov = curve_fit(pv_v2_func, qcut, sqcut, p0=first_guess,
                           bounds=([1,0,.000001,-100,-100,-100],[30,1,10,100,100,100]))
    return np.array(pv_v2_func(q,*popt))


def sum_scans_rng(df_sq, scan_min, scan_max, df_pc):
    avg_q = np.array(df_sq.index)
    avg_sq = np.zeros(len(avg_q))

    pc_sum = 0.0
    for i in range(len(df_sq.columns)):
        col = df_sq.columns[i]
        if col >= scan_min and col <= scan_max:
            pc_sum += df_pc.loc['pc',col]

    print ('pc sum from that set : '+str(pc_sum))
    for i in range(len(df_sq.columns)):
        col = df_sq.columns[i]
        if col >= scan_min and col <= scan_max:
            this_pc = df_pc.loc['pc',col]
            avg_sq += df_sq.loc[:,col]*(this_pc/pc_sum)

    return np.array(avg_q), np.array(avg_sq)


def sum_scans_list(df_sq, scan_list, df_pc):
    avg_q = np.array(df_sq.index)
    avg_sq = np.zeros(len(avg_q))

    pc_sum = 0.0
    for i in range(len(df_sq.columns)):
        col = df_sq.columns[i]
        if col in scan_list:
            pc_sum += df_pc.loc['pc',col]

    print ('pc sum from that set : '+str(pc_sum))
    for i in range(len(df_sq.columns)):
        col = df_sq.columns[i]
        if col in scan_list:
            this_pc = df_pc.loc['pc',col]
            avg_sq += df_sq.loc[:,col]*(this_pc/pc_sum)

    return np.array(avg_q), np.array(avg_sq)


def fit_ndeg_to_sq(q,sq,qmin=10,qmax=31,ndeg=2):
    qcut, sqcut = cut_data(q,sq,qmin,qmax)

    this_pfit = np.poly1d(np.polyfit(qcut,sqcut,ndeg))
    sqfit = this_pfit(q)
    return sqfit


def  showme_reduction_fgbg_gr(q, _sqfg, _sqbg, rmin=.5,rmax=20,delr=.02,v_grmin=-13,v_grmax=13):

    def f1(bgd_scaler,gauss_damp,gw,uqmax,uqmin,vis_bg_yadjust):
        if gauss_damp:
            r,bggr = make_gr_from_sq(q,(bgd_scaler*_sqbg)*gauss(q,gw,0),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)
        else:
            r,bggr = make_gr_from_sq(q,(bgd_scaler*_sqbg),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)

        if gauss_damp:
            r,fggr = make_gr_from_sq(q,(_sqfg)*gauss(q,gw,0),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)
        else:
            r,fggr = make_gr_from_sq(q,(_sqfg),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)
        gr = fggr - bggr

        plt.figure(figsize=(12,6))

        ax1=plt.subplot(211)
        plt.plot(r,fggr,'b')
        plt.plot(r,bggr+vis_bg_yadjust,'r')

        plt.xlim([rmin,rmax])
        plt.xlabel('r')
        plt.ylabel('G(r)')

        plt.subplot(212,sharey=ax1)
        plt.plot(r,gr,'k',label='D2')
        plt.xlim([rmin,rmax])
        plt.xlabel('r')
        plt.ylabel('G(r)')

    do_cont_update = widgets.Checkbox(value=True,description='Continuous Update')

    bgd_scaler=widgets.FloatSlider(min=-1.0,max=2.0,step=.01,value=1.0,continuous_update=do_cont_update.value,description='Rescale Bgd')
    gauss_damp=widgets.Cheox(value=False,description='Apply Damping to S(Q)')
    gw=widgets.FloatSlider(min=0.01,meax=30.0,step=.1,value=30.0,continuous_update=do_cont_update.value,description='Gauss Wid')
    uqmax=widgets.FloatSlider(min=0.0,max=30.0,step=.1,value=30.0,continuous_update=do_cont_update.value,description='Qmax')
    uqmin=widgets.FloatSlider(min=0,max=5,step=.1,value=1.0,continuous_update=do_cont_update.value,description='Qmin')
    vis_bg_yadjust=widgets.FloatSlider(min=-20.,max=10.0,step=.1,value=0.0,continuous_update=do_cont_update.value,description='Bgd Offset')


    def update_const_update(*args):
        bgd_scaler.continuous_update = do_cont_update.value
        gw.continuous_update = do_cont_update.value
        uqmax.continuous_update = do_cont_update.value
        uqmin.continuous_update = do_cont_update.value
        vis_bg_yadjust.continuous_update = do_cont_update.value

    do_cont_update.observe(update_const_update, 'value')

    row1_ui = widgets.HBox([bgd_scaler])
    row2_ui = widgets.HBox([gauss_damp,gw])
    row3_ui = widgets.HBox([uqmin, uqmax])
    row4_ui = widgets.HBox([vis_bg_yadjust,do_cont_update])

    ui = widgets.VBox([row1_ui, row2_ui, row3_ui, row4_ui])

    out = widgets.interactive_output(f1, {'bgd_scaler':bgd_scaler,'gauss_damp':gauss_damp,'gw':gw,'uqmax':uqmax,'uqmin':uqmin,'vis_bg_yadjust':vis_bg_yadjust})

    display(ui, out,flex='flex-grow')


def showme_reduction_sq_and_gr(q,_sqfg,_sqbg,v_qmin=0.0,v_qmax=30,v_imin=-2,v_imax=12,
                                    rmin=.5,rmax=20,delr=.02,v_grmin=-13,v_grmax=13):


    def f3(bgd_scaler,gauss_damp,gw,uqmax,uqmin,show_fq):

        if gauss_damp:
            r,gr = make_gr_from_sq(q,(_sqfg-bgd_scaler*_sqbg)*gauss(q,gw,0),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)
        else:
            r,gr = make_gr_from_sq(q,(_sqfg-bgd_scaler*_sqbg),qmin=uqmin,
                                   qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)

        plt.figure(figsize=(12,6))
        plt.subplot(121)

        if gauss_damp==False:
            ugw = 9999.9
        else:
            ugw = gw
            plt.plot(q,gauss(q,ugw,0),'r--')

        if show_fq == False:
            plt.plot(q,_sqfg*gauss(q,ugw,0),'k')
            plt.plot(q,bgd_scaler*_sqbg*gauss(q,ugw,0),'b')
        else:
            plt.plot(q,q*_sqfg*gauss(q,ugw,0),'k')
            plt.plot(q,q*bgd_scaler*_sqbg*gauss(q,ugw,0),'b')
            plt.plot([v_qmin,v_qmax],[0,0],'k')

        plt.axis([v_qmin,v_qmax,v_imin,v_imax])

        plt.xlabel('Q')
        if show_fq:
            plt.ylabel('F(Q)')
            plt.autoscale(enable=True,axis='y',tight=None)

        else:
            plt.ylabel('S(Q)')

        cymin, cymax = plt.ylim()
        plt.plot([uqmax,uqmax],[cymin,cymax],color='purple',ls='--')
        plt.plot([uqmin,uqmin],[cymin,cymax],color='purple',ls='--')
        plt.ylim(cymin,cymax)

        plt.subplot(122)
        plt.plot(r,gr,'k',label='D2')
        plt.axis([rmin,rmax,v_grmin,v_grmax])
        plt.xlabel('r')
        plt.ylabel('G(r)')
    #now to setup GUI and widgets

    do_cont_update = widgets.Checkbox(value=True,description='Continuous Update')

    bgd_scaler=widgets.FloatSlider(min=-1.0,max=2.0,step=.01,value=1.0,continuous_update=do_cont_update.value,description='Rescale Bgd')
    gauss_damp=widgets.Checkbox(value=False,description='Apply Damping to S(Q)')
    gw=widgets.FloatSlider(min=0.01,meax=30.0,step=.1,value=30.0,continuous_update=do_cont_update.value,description='Gauss Wid')
    uqmax=widgets.FloatSlider(min=0.0,max=30.0,step=.1,value=30.0,continuous_update=do_cont_update.value,description='Qmax')
    uqmin=widgets.FloatSlider(min=0,max=30.0,step=.1,value=1.0,continuous_update=do_cont_update.value,description='Qmin')
    show_fq = widgets.Checkbox(value=False,description='Display F(Q)')


    def update_const_update(*args):
        bgd_scaler.continuous_update = do_cont_update.value
        gw.continuous_update = do_cont_update.value
        uqmax.continuous_update = do_cont_update.value
        uqmin.continuous_update = do_cont_update.value


    def update_qmin_limits(*args):
        uqmin.max = uqmax.value

    do_cont_update.observe(update_const_update, 'value')
    uqmax.observe(update_qmin_limits, 'value')

    row1_ui = widgets.HBox([bgd_scaler])
    row2_ui = widgets.HBox([gauss_damp,gw])
    row3_ui = widgets.HBox([uqmin, uqmax])
    row4_ui = widgets.HBox([show_fq, do_cont_update])

    ui = widgets.VBox([row1_ui, row2_ui, row3_ui, row4_ui])


    out = widgets.interactive_output(f3, {'bgd_scaler':bgd_scaler,'gauss_damp':gauss_damp,'gw':gw,'uqmax':uqmax,'uqmin':uqmin,'show_fq':show_fq})

    display(ui, out,flex='flex-grow')


def do_reduction_placzek_corrections(q,sqfg,bgd,rescale_bgd=1.0,plaz_type=None,
                                     gauss_damp=False,gw=20.0,qmax=None,qmin=None,
                                     rmin=0.0,rmax=20.0,delr=.02
                                     ,qminpla=10.0,qmaxpla=30.0,ndeg=2, return_correction = False,
                                    skip_bgd = False, return_final_sq = False, force_qmax_type='Off'):
    """
    Perform r,PDF data reduction using the full suite of options available in the 'showme_reduction_placzek_corrections' widget.

    Must pass q, sq(foreground), and sq(background).  Futher options listed below (with default values).

    rescale_bgd = Scaling factor for background (1.0)

    plaz_type = Method of correcting high-Q behavior, approximating a Placzek correction (None)
        'ndeg' = polynomial correction
        'pv' = Pseudo-Voight correction
        'pvndeg0' = Pseudo-Voight + 0th order polynomial, which is just a constant
        'pvndeg1' = Pseudo-Voight + 1st order polynomial
        'pvndeg2' = Pseudo-Voight + 2nd order polynomial
        'None' = skip this correction

    gauss_damp = Apply a gaussian damping envelope to the S(Q) data prior to transform (False)

    gw = If gauss_damp is applied, the width of the Gaussian used in Q (20.0)

    qmax = Value of Qmax to use in generation of PDF (default None=native Qmax of data)

    qmin = Value of Qmin to use in generation of PDF (default None=native Qmin of data)

    rmin = Lowest r-value calculated in the returned PDF (0.0)

    rmax = Highest r-value calculated in the returned PDF (20.0)

    delr = Spacing to use in r-binning of calculated PDF (0.02)

    qminpla = Lower bound of range used to fit Placzek correction (10.0)

    qmaxpla = Upper bound of range used to fit Placzek correction (30.0)

    ndeg = If using polynomial as plaz_type, this is the degree of polynomial useed (2)

    return_correction = Instead of returning the calculated r/G(r), returns the calculated Placzek correction (False)

    return_final_sq = Instead of returning the calculated r/G(r), returns the fully corrected S(Q) (False)

    force_qmax_type = Forces the final S(Q) to terminate at S(Qmax)=0.  Can help reduce high-frequency noise in PDF (Off)

    """
    #first, make netsq if bgd and/or damping is present
    q = np.array(q)
    sqfg = np.array(sqfg)
    bgd = np.array(bgd)

    if skip_bgd:
        netsq = sqfg
    else:
        netsq = sqfg - bgd*rescale_bgd


    if gauss_damp:
        netsq = netsq*gauss(q,gw,0)


    if force_qmax_type == 'Force Data (PreCorrection)':
        qcut, sqcut = cut_data(q,netsq,qmax-.5,qmax)
        mean_sqmax = np.mean(sqcut)
        netsq -= mean_sqmax

    #now, apply a correction if requested
    if plaz_type != None:
        if plaz_type == 'Polynomial' or plaz_type == 'poly' or plaz_type == 'ndeg':
            sq_poly_fit = fit_ndeg_to_sq(q,netsq,ndeg=ndeg,qmin=qminpla,qmax=qmaxpla)
            this_fit = sq_poly_fit
        elif plaz_type == 'Pseudo-Voight' or plaz_type == 'pv' or plaz_type == 'hydro':
            pv_fit = fit_pv_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_fit
        elif plaz_type == 'PVoight + n0' or plaz_type == 'pvndeg0':
            pv_n0_fit = fit_pv_n0_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n0_fit
        elif plaz_type == 'PVoight + n1' or plaz_type == 'pvndeg1':
            pv_n1_fit = fit_pv_n1_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n1_fit
        elif plaz_type == 'PVoight + n2' or plaz_type == 'pvndeg2':
            pv_n2_fit = fit_pv_n2_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n2_fit
        else:
            print ("I don't know that correction type, sorry")
            this_fit = np.zeros(len(q))
    else:
        this_fit = np.zeros(len(q))

    if force_qmax_type == 'Force Data' or force_qmax_type == 'Force Both (Independent)':
        qcut, sqcut = cut_data(q,netsq,qmax-.5,qmax)
        mean_sqmax = np.mean(sqcut)
        netsq -= mean_sqmax
    if force_qmax_type == 'Force Correction' or force_qmax_type == 'Force Both (Independent)':
        qcut, sqcut = cut_data(q,this_fit,qmax-.5,qmax)
        mean_sqmax = np.mean(sqcut)
        this_fit -= mean_sqmax
    if force_qmax_type == 'ReCorrection':
        qcut, sqcut = cut_data(q,netsq-this_fit,qmax-.5,qmax)
        mean_sqmax = np.mean(sqcut)
        this_fit += mean_sqmax

    netsq = netsq - this_fit

    if return_correction:
        return this_fit

    if return_final_sq:
        return netsq

    #finally, generate PDF
    r,gr = make_gr_from_sq(q,netsq,qmin=qmin,qmax=qmax,rmin=rmin,rmax=rmax,delr=delr)

    return r,gr


def showme_reduction_placzek_corrections(q,_sqfg,_sqbg,v_qmin=0.0,v_qmax=30,v_imin=-2,v_imax=12,
                                    rmin=.5,rmax=20,delr=.02,v_grmin=-13,v_grmax=13,figsize=(12,8)):
    q = np.array(q)
    _sqfg = np.array(_sqfg)
    _sqbg = np.array(_sqbg)


    def f2(bgd_scaler,gauss_damp,gw,uqmax,uqmin,show_fq,ndeg,qminpla,qmaxpla,plaz_type,altplot_type,
           vis_bg_yadjust,force_qmax_type):
        #use_ndeg = False

        plt.figure(figsize=figsize)
        ax1=plt.subplot(221)

        if gauss_damp==False:
            ugw = 9999.9
        else:
            ugw = gw

        netsq = (_sqfg - bgd_scaler*_sqbg)*gauss(q,ugw,0)

        if force_qmax_type == 'Force Data (PreCorrection)':
            qcut, sqcut = cut_data(q,netsq,uqmax-.5,uqmax)
            mean_sqmax = np.mean(sqcut)
            netsq -= mean_sqmax

        if plaz_type == 'Polynomial':
            sq_poly_fit = fit_ndeg_to_sq(q,netsq,ndeg=ndeg,qmin=qminpla,qmax=qmaxpla)
            this_fit = sq_poly_fit
        elif plaz_type == 'Pseudo-Voight':
            pv_fit = fit_pv_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_fit
        elif plaz_type == 'PVoight + n0':
            pv_n0_fit = fit_pv_n0_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n0_fit
        elif plaz_type == 'PVoight + n1':
            pv_n1_fit = fit_pv_n1_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n1_fit
        elif plaz_type == 'PVoight + n2':
            pv_n2_fit = fit_pv_n2_to_sq(q,netsq,qmin=qminpla,qmax=qmaxpla)
            this_fit = pv_n2_fit

        if force_qmax_type == 'Force Data' or force_qmax_type == 'Force Both (Independent)':
            qcut, sqcut = cut_data(q,netsq,uqmax-.5,uqmax)
            mean_sqmax = np.mean(sqcut)
            netsq -= mean_sqmax
        if force_qmax_type == 'Force Correction' or force_qmax_type == 'Force Both (Independent)':
            qcut, sqcut = cut_data(q,this_fit,uqmax-.5,uqmax)
            mean_sqmax = np.mean(sqcut)
            this_fit -= mean_sqmax
        if force_qmax_type == 'ReCorrection':
            qcut, sqcut = cut_data(q,netsq-this_fit,uqmax-.5,uqmax)
            mean_sqmax = np.mean(sqcut)
            this_fit += mean_sqmax

        if show_fq == False:
            plt.plot(q,netsq,'k')
            plt.plot(q,this_fit,'r')

        else:
            plt.plot(q,q*netsq,'k')
            plt.plot(q,q*this_fit,'r')

        cymin, cymax = plt.ylim()
        plt.plot([qminpla,qminpla],[cymin,cymax],color='green',ls='--')
        plt.plot([qmaxpla,qmaxpla],[cymin,cymax],color='purple',ls='--')
        plt.ylim(cymin,cymax)

        #plt.axis([v_qmin,v_qmax,v_imin,v_imax])
        plt.xlabel('Q')
        if show_fq:
            plt.ylabel('F(Q)')
            plt.autoscale(enable=True,axis='y',tight=None)
        else:
            plt.ylabel('S(Q)')

        if show_qminqmax.value == True:
            cymin, cymax = plt.ylim()
            plt.plot([uqmax,uqmax],[cymin,cymax],color='red',ls='--',alpha=.8)
            plt.plot([uqmin,uqmin],[cymin,cymax],color='red',ls='--',alpha=.8)
            plt.ylim(cymin,cymax)

        plt.xlim(min(q),max(q))

        r,gr_correction = make_gr_from_sq(q,this_fit,qmin=uqmin,
                        qmax=uqmax, rmin=rmin, rmax=rmax, delr=delr)
        r,gr_raw = make_gr_from_sq(q,netsq,qmin=uqmin,
                        qmax=uqmax,rmin=rmin,rmax=rmax,delr=delr)

        plt.subplot(222)

        if altplot_type=='Zoomed Fit':

            if show_fq:
                plt.plot(q,q*netsq,'k')
                plt.xlim(qminpla,qmaxpla)
                plt.plot(q,q*this_fit,'r')
                cutq,cutsq = cut_data(q,netsq,qminpla,qmaxpla)
                plt.ylim(min(cutq*cutsq),max(cutq*cutsq))
                plt.xlim(min(cutq),max(cutq))
            else:
                plt.plot(q,netsq,'k')
                plt.xlim(qminpla,qmaxpla)
                plt.plot(q,this_fit,'r')
                cutq,cutsq = cut_data(q,netsq,qminpla,qmaxpla)
                plt.ylim(min(cutsq),max(cutsq))

        elif altplot_type=='Final SQ':
            if show_fq:
                plt.plot(q,q*(netsq-this_fit),'b')
                cymin, cymax = plt.ylim()

                plt.ylabel('F(Q) [Final]')
                plt.plot([0,30],[0,0],'k')
                plt.ylim(cymin, cymax)
                plt.xlim(uqmin,uqmax)
            else:

                plt.plot(q,netsq-this_fit,'b')
                cymin, cymax = plt.ylim()

                plt.ylabel('S(Q) [Final]')
                plt.xlim(uqmin,uqmax)
                plt.ylim(cymin, cymax)

        elif altplot_type =='FG/BG S(Q)':
            if show_fq:
                plt.plot(q,q*_sqfg*gauss(q,ugw,0),'purple')
                plt.plot(q,q*bgd_scaler*_sqbg*gauss(q,ugw,0),'orange')
                plt.xlim(min(q),max(q))

            else:
                plt.plot(q,_sqfg*gauss(q,ugw,0),'purple')
                plt.plot(q,bgd_scaler*_sqbg*gauss(q,ugw,0),'orange')
                plt.xlim(min(q),max(q))

        elif altplot_type == 'FG/BG G(r)':
            rt,gr_fg = make_gr_from_sq(q,_sqfg,qmin=uqmin,qmax=uqmax,rmin=0.80,rmax=5.0,delr=delr)
            rt,gr_bg = make_gr_from_sq(q,_sqbg*bgd_scaler,qmin=uqmin,qmax=uqmax,rmin=0.80,rmax=5.0,delr=delr)

            plt.plot(rt, gr_fg)
            plt.plot(rt, gr_bg+vis_bg_yadjust)

        plt.xlabel('Q')

        plt.subplot(223)

        plt.plot(r,gr_raw,'k')
        plt.plot(r,gr_correction,'r')
        plt.xlim(min(r),max(r))
        plt.xlabel('r')
        plt.ylabel('G(r)')

        plt.subplot(224)

        gr = gr_raw-gr_correction
        plt.plot(r,gr,'b')
        plt.xlim(min(r),max(r))
        plt.xlabel('r')
        plt.ylabel('G(r) [final]')
    #### Setup the GUI (gw,uqmax,uqmin,show_fq,ndeg,qminpla,qmaxpla,show_zoomed_fit)

    do_cont_update = widgets.Checkbox(value=True,description='Continuous Update')

    bgd_scaler=widgets.FloatSlider(min=-1.0,max=2.0,step=.01,value=1.0,continuous_update=do_cont_update.value,
                                   description='Rescale Bgd')
    gauss_damp=widgets.Checkbox(value=False,description='Apply Damping to S(Q)')
    gw=widgets.FloatSlider(min=0.01,meax=30.0,step=.1,value=30.0,continuous_update=do_cont_update.value,description='Gauss Wid')
    uqmax=widgets.FloatSlider(min=0.0,max=q[-1],step=.1,value=30.0,continuous_update=do_cont_update.value,description='Qmax')
    uqmin=widgets.FloatSlider(min=0,max=q[-1],step=.1,value=1.0,continuous_update=do_cont_update.value,description='Qmin')
    show_fq = widgets.Checkbox(value=False,description='Display F(Q)')
    ndeg = widgets.IntSlider(min=0,max=8,step=1,value=0,continuous_update=do_cont_update.value,description='Ndeg Value')
    qminpla=widgets.FloatSlider(min=0.0,max=q[-1],step=.1,value=10.0,continuous_update=do_cont_update.value,
                                description='QMinPla')
    qmaxpla=widgets.FloatSlider(min=0.0,max=q[-1],step=.1,value=30.0,continuous_update=do_cont_update.value,
                                description='QMaxPla')
    show_qminqmax = widgets.Checkbox(value=False,description='Show QMin/QMax')
    plaz_type = widgets.Dropdown(options=['Polynomial','Pseudo-Voight','PVoight + n0','PVoight + n1','PVoight + n2'],
                                 value='Polynomial',description='Correction Type')
    altplot_type = widgets.Dropdown(options=['Final SQ','Zoomed Fit','FG/BG S(Q)','FG/BG G(r)'],
                                    value='Final SQ',description='Alt-Plot')
    vis_bg_yadjust=widgets.FloatSlider(min=-10.,max=10.0,step=.1,value=0.0,continuous_update=do_cont_update.value,
                                       description='Bgd Offset')
    force_qmax_at_zero = widgets.Checkbox(value=False,description='Force S(Qmax)->0')
    force_qmax_type = widgets.Dropdown(options=['Off','ReCorrection','Force Both (Independent)',
                                                'Force Data (PreCorrection)','Force Data','Force Correction'],
                                       value='Off',description='S(Qmax)->0')


    def update_const_update(*args):
        bgd_scaler.continuous_update = do_cont_update.value
        gw.continuous_update = do_cont_update.value
        uqmax.continuous_update = do_cont_update.value
        uqmin.continuous_update = do_cont_update.value
        ndeg.continuous_update = do_cont_update.value
        qminpla.continuous_update = do_cont_update.value
        qmaxpla.continuous_update = do_cont_update.value
        vis_bg_yadjust.continuous_update = do_cont_update.value


    def update_qmin_limits(*args):
        qminpla.max = qmaxpla.value
        uqmin.max = uqmax.value


    def update_correction_type(*args):
        if plaz_type.value == 'Polynomial':
            #qminpla.value = 10.0
            #qmaxpla.value = 30.0
            ndeg.disabled = False
        else: # plaz_type.value == 'Pseudo-Voight':
            #qminpla.value = 1.0
            #qmaxpla.value = 30.0
            ndeg.disabled = True
            if plaz_type.value == 'PVoight + n0':
                ndeg.value = 0
            elif plaz_type.value == 'PVoight + n1':
                ndeg.value = 1
            elif plaz_type.value == 'PVoight + n2':
                ndeg.value = 2
        #elif plaz_type.value == 'PVoight + n1':
        #    qminpla.value = 1.0
        #    qmaxpla.value = 30.0

    plaz_type.observe(update_correction_type,'value')
    do_cont_update.observe(update_const_update, 'value')
    qmaxpla.observe(update_qmin_limits, 'value')
    uqmax.observe(update_qmin_limits, 'value')

    row1_ui = widgets.HBox([bgd_scaler,plaz_type,do_cont_update])
    #row1_ui = widgets.HBox([bgd_scaler])

    row2_ui = widgets.HBox([gauss_damp,gw,vis_bg_yadjust])

    row3_ui = widgets.HBox([uqmin, uqmax, force_qmax_type])
    row4_ui = widgets.HBox([show_fq,altplot_type,show_qminqmax])
    row5_ui = widgets.HBox([ndeg,qminpla,qmaxpla])

    ui = widgets.VBox([row1_ui, row2_ui, row3_ui, row4_ui, row5_ui])


    out = widgets.interactive_output(f2, {'bgd_scaler':bgd_scaler,'gauss_damp':gauss_damp,'gw':gw,'uqmax':uqmax,'uqmin':uqmin,
                                          'qminpla':qminpla,'qmaxpla':qmaxpla,'ndeg':ndeg,'show_fq':show_fq,
                                          'plaz_type':plaz_type,'altplot_type':altplot_type,'vis_bg_yadjust':vis_bg_yadjust,
                                         'force_qmax_type':force_qmax_type})

    display(ui, out,flex='flex-grow')


def make_gr_from_sq(q, sq, delr=None, rmax = None, rmin=None,qmin = None, qmax=None, return_qsq = False,correct_for_qmax=False,final_wid = 10):
    if delr == None:
        delr = np.pi/q[-1]
    if rmax == None:
        rmax = np.pi/(q[1]-q[0])
    if rmin == None:
        rmin = 0
    if qmin == None:
        qmin = q[0]
    if qmax == None:
        qmax = q[-1]

    selected_mask = np.where(np.logical_and(q>=qmin, q<=qmax))
    useq = q[selected_mask]
    usesq = sq[selected_mask]

    if correct_for_qmax:
        correct_val = usesq[-final_wid:-1].mean()
        usesq -= correct_val

    if return_qsq:
        return useq, usesq

    r = np.arange(rmin,rmax+delr/2,delr)
    r, gr = pdf_transform(r[0],r[-1],delr, useq, usesq)
    return r, gr


def debye_scattering(r_list,gr_list,q_list):
    sq_list = np.zeros(len(q_list))
    for j in range(len(q_list)):
        q = q_list[j]
        sq_list[j] = np.sum(gr_list*np.sin(q*r_list))
    for i in range(len(q_list)):
        if q_list[i] > 0:
            sq_list[i] = sq_list[i]/q_list[i]
        else:
            sq_list[i] = 0.0
    return sq_list


def pdf_transform(xmin,xmax,delx,x,y):
    r = np.arange(xmin,xmax+delx/2.0,delx)
    gr = np.zeros(len(r))
    for rvals in range(len(r)):
        gr[rvals] = np.sum( x * y * np.sin(r[rvals]*x) )
        #for qvals in range(len(x)):
        #    gr[rvals] += x[qvals]*y[qvals]*np.sin(r[rvals]*x[qvals])
    gr = gr / (32.0)
    return r,gr


def make_sq_from_gr(r,gr,qmax=None,delq=None,qmin=None,rmin=None,rmax=None):
    if delq==None:
        delq = np.pi/r[-1]
        print ('using default Nyquist delq '+str(delq))
    if qmax==None:
        qmax = np.pi/(r[1]-r[0])
        print ('using default Nyquist qmax '+str(qmax))
    if qmin == None:
        qmin = 0.0
    q = np.arange(qmin,qmax+delq/2,delq)

    if rmax == None:
        rmax = r[-1]
    if rmin == None:
        rmin = r[0]

    selected_mask = np.where(np.logical_and(r>=rmin, r<=rmax))
    user = r[selected_mask]
    usegr = gr[selected_mask]

    usegr = np.nan_to_num(usegr)
    sq = debye_scattering(user, usegr, q)/(5*np.pi)


    sq *= 2./np.pi

    sq = sq * 25.0 / (len(r)*(q[1]-q[0]))
    return q,np.nan_to_num(sq)

def gauss(x,w,cen):
    return np.exp(-((x-cen)**2.)/(2.*(w**2)))

def lorz(x,w):
    return np.exp(-(x)/(2.*(w)))


def read_pdfgui_gr(filename,junk=3,backjunk = 2):
    with open(filename,'r') as infile:
        datain = infile.readlines()
    if backjunk == 0:
        datain = datain[junk:]
    else:
        datain = datain[junk:-backjunk]

    xin = np.zeros(len(datain))
    yin = np.zeros(len(datain))

    print ('length '+str(len(xin)))
    for i in range(len(datain)):
        xin[i]= float(datain[i].split()[0])
        yin[i]= float(datain[i].split()[1])
    return xin,yin

def my_first_convolution_gauss(rlist,data,wid,use_delq_over_q=False,dqoq_power = 1):
    new_data = np.zeros(len(data))
    for i in range(len(rlist)):
        r = rlist[i]
        if use_delq_over_q:
            use_wid = wid*r**dqoq_power
        else:
            use_wid = wid
        #now build up the gaussian profile
        weights = gauss(rlist, use_wid, r)
        weights = weights / sum(weights)
        new_data[i] = sum(weights * data)

    return new_data


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]

def top_down_plot(data,**kwargs):
    """plots 2D data from a pandas dataframe.  Requires that both index and column values of data frame be numeric!
    Optional arguments are colormap minimum and maximum (cmin, cmax), colormap to use (cmap), x-min and x-max
    (xmin, xmax), y-value min and max (ymin, ymax),  if you want gouraud-style blurring (blur = True, default False),
    and if you want a colorbar plotted (colorbar = True, default False)"""

    X,Y = np.meshgrid(data.index,data.columns)
    X=X.T
    Y=Y.T

    cmin = min(data.min())
    cmax = max(data.max())
    use_cmap = 'viridis'
    xmin = min(data.index)
    xmax = max(data.index)
    ymin = min(data.columns)
    ymax = max(data.columns)
    shading_choice = 'None'
    use_colorbar = False

    if kwargs is not None:
        for key, value in kwargs.items():
            #print ("%s set to %s" %(key,value))
            if key == 'xmin':
                xmin = value
            if key == 'xmax':
                xmax = value
            if key == 'ymax':
                ymax = value
            if key == 'ymin':
                ymin = value
            if key == 'cmin':
                cmin = value
            if key == 'cmax':
                cmax = value
            if key == 'cmap':
                use_cmap = value
            if key == 'blur':
                if value == True:
                    shading_choice = 'gouraud'
                else:
                    shading_choice = 'None'
            if key == 'colorbar':
                if value == True:
                    use_colorbar = True
    plt.plot
    plt.pcolormesh(X,Y,data,cmap=use_cmap,vmin = cmin, vmax = cmax, shading = shading_choice)
    plt.axis([xmin,xmax,ymin,ymax])
    if use_colorbar:
        plt.colorbar()
    plt.show()

def is_number(arg1):
    if len(arg1.strip())>0:
        try:
            float(arg1)
            return True
        except ValueError:
            return False

def calc_rw(indata,infit,return_crw = False):
    fit = np.array(infit)
    data = np.array(indata)

    if return_crw == False:
        top = sum((data-fit)**2.0)
        bottom = sum(data**2.0)
        if bottom != 0.0:
            return ((top/bottom)**0.5)
        else:
            return 0.0
    else:
        vals = np.zeros(len(data))
        tdata_sum = 0.0
        tdiff_sum = 0.0
        for i in range(len(data)):
            tdata_sum += (data[i])**2
            tdiff_sum += (data[i] - fit[i])**2
            if tdata_sum != 0.0:
                vals[i] = tdiff_sum/tdata_sum
            else:
                vals[i] = 0.0
        vals = vals**(0.5)
        return vals


def calc_residual(y1,y2,return_sum = True,return_abs = True):
    res = np.zeros(len(y1))
    if  return_abs == False:
        res = (y1 - y2)
    if return_abs == True:
        res = abs(y1 - y2)

    if return_sum:
        return res.sum()
    else:
        return res

def cut_data(qt,sqt,qmin,qmax):
    qt_back, sqt_back = qt[qt > qmin], sqt[qt > qmin]
    qt_back, sqt_back = qt_back[qt_back < qmax], sqt_back[qt_back < qmax]
    return qt_back, sqt_back

def read_twocol_data(filename,junk=0,backjunk = 0, splitchar=None, do_not_float=False, shh=False):
    with open(filename,'r') as infile:
        datain = infile.readlines()
    if backjunk == 0:
        datain = datain[junk:]
    else:
        datain = datain[junk:-backjunk]

    xin = np.zeros(len(datain))
    yin = np.zeros(len(datain))

    if shh == False:
        print ('length '+str(len(xin)))
    if do_not_float:
        if splitchar==None:
            for i in range(len(datain)):
                xin[i]= (datain[i].split()[0])
                yin[i]= (datain[i].split()[1])
        else:
            for i in range(len(datain)):
                xin[i]= (datain[i].split(splitchar)[0])
                yin[i]= (datain[i].split(splitchar)[1])
    else:
        if splitchar==None:
            for i in range(len(datain)):
                xin[i]= float(datain[i].split()[0])
                yin[i]= float(datain[i].split()[1])
        else:
            for i in range(len(datain)):
                xin[i]= float(datain[i].split(splitchar)[0])
                yin[i]= float(datain[i].split(splitchar)[1])

    return xin,yin

def similarity_matrix(df,use_abs=True,rmin=0,rmax=100):
    score_matrix = np.zeros([len(df.columns),len(df.columns)])

    for y1i in range(len(df.columns)):
        col1 = df.columns[y1i]
        for y2i in range(len(df.columns)):
            col2 = df.columns[y2i]
            score_matrix[y1i,y2i] = calc_residual(df.loc[rmin:rmax,col1],df.loc[rmin:rmax,col2],return_sum=True,return_abs=use_abs)

    df_score = pd.DataFrame(data = score_matrix, index=df.columns, columns= df.columns)

    return df_score

def single_parent_difference(mom,df,return_sum=True,abs_score = True):
    score_matrix = np.zeros([len(df.index),len(df.columns)])
    res_array = df.copy(deep=True)
    for cols in res_array.columns:
        res_array[cols] = df[cols]-mom

    if abs_score:
        res_array = res_array.abs()

    if return_sum:
        return res_array.sum()
    else:
        return res_array

def two_parent_combo(mom,mom_frac,dad,dad_frac):
    return mom*mom_frac + dad*dad_frac

def solve_parent_fraction(mom,dad,df):

    def two_parent_simple_combo(r,mom_frac):
        return mom*mom_frac + dad * (1-mom_frac)

    popt, pcov = curve_fit(two_parent_simple_combo, df.index, df.values, p0 = (.5))

    return popt[0]

def try_all_the_things(mom,dad,df,num_pts=101,confidence = 0.05,return_error_bars = True,return_full_res_map = False):

    df_phi_score = pd.DataFrame(index=np.linspace(0,1,num_pts),columns=df.columns,data=0)


    best_phi_list = []
    best_res_list = []

    if confidence > 1.0:
        confidence *= .01

    for cols in df_phi_score.columns:

        best_phi = 0
        best_res = 1e15

        test_data = df.loc[:,cols]
        for rows in df_phi_score.index:
            fit_frac = rows
            this_res = calc_residual(two_parent_combo(mom,fit_frac,dad,1-fit_frac),test_data)

            if this_res < best_res:
                best_res = this_res
                best_phi = fit_frac

            df_phi_score.loc[rows,cols] = this_res

        best_phi_list.append(best_phi)
        best_res_list.append(best_res)

    best_phi_list = np.array(best_phi_list)
    best_res_list = np.array(best_res_list)

    wbp_low_list = []
    wbp_high_list = []

    if return_error_bars:
        for i in range(len(df_phi_score.columns)):
            cols = df.columns[i]
            worst_best_res = (1.0+confidence) * best_res_list[i]

            wbp_low = 0
            wbp_high = 1

            for j in range(0,len(df_phi_score.index),1):
                rows = df_phi_score.index[j]
                if df_phi_score.loc[rows,cols] <= worst_best_res:
                    wbp_low = rows
                    break

            for j in range(len(df_phi_score.index)-1,-1,-1):
                rows = df_phi_score.index[j]
                if df_phi_score.loc[rows,cols] <= worst_best_res:
                    wbp_high = rows
                    break

            wbp_high_list.append(wbp_high)
            wbp_low_list.append(wbp_low)



        wbp_high_list = np.array(wbp_high_list)
        wbp_low_list = np.array(wbp_low_list)

        if return_full_res_map == False:
            return best_phi_list, best_res_list, wbp_high_list, wbp_low_list
        else : #return the full res_map
            return best_phi_list, best_res_list, wbp_high_list, wbp_low_list,df_phi_score

    else : #don't bother with error bars
        if return_full_res_map == False:
            return best_phi_list, best_res_list
        else : #return the full res_map
            return best_phi_list,best_res,df_phi_score

def write_out_file(filename,x,y):
    outf = open(filename,'w')
    for i in range(len(x)):
        outf.write(str(x[i])+' '+str(y[i])+'\n')
    outf.close()

def make_df_full_set(file_preface,tstart = 19,tend = 22):
    file_list = glob.glob(file_preface+'*')

    tlist = np.zeros(len(file_list))
    for i in range(len(file_list)):
        tlist[i] = float(file_list[i][tstart:tend])

    r,gr = read_twocol_data(file_list[0],junk=5,shh=True)

    this_df = pd.DataFrame(index=r)
    this_df[tlist[0]] = gr

    for i in range(1,len(tlist)):
        r,gr = read_twocol_data(file_list[i],junk=5,shh=True)
        this_df[tlist[i]] = gr

    return this_df

def read_index_data_smart(filename,junk=None,backjunk=None,splitchar=None, do_not_float=False, shh=True, use_idex=[0,1]):
    with open(filename,'r') as infile:
        datain = infile.readlines()

    if junk == None:
        for i in range(len(datain)):
            try:
                for j in range(10):
                    x1,y1 = float(datain[i+j].split(splitchar)[use_idex[0]]), float(datain[i+j].split(splitchar)[use_idex[1]])
                junk = i
                break
            except:
                pass #print ('nope')

    if backjunk == None:
        for i in range(len(datain),-1,-1):
            try:
                x1,y1 = float(datain[i].split(splitchar)[use_idex[0]]), float(datain[i].split(splitchar)[use_idex[1]])
                backjunk = len(datain)-i-1
                break
            except:
                pass
                #print ('nope')

    #print ('found junk '+str(junk))
    #print ('and back junk '+str(backjunk))

    if backjunk == 0:
        datain = datain[junk:]
    else:
        datain = datain[junk:-backjunk]

    xin = np.zeros(len(datain))
    yin = np.zeros(len(datain))

    if do_not_float:
        xin = []
        yin = []

    if shh == False:
        print ('length '+str(len(xin)))
    if do_not_float:
        if splitchar==None:
            for i in range(len(datain)):
                xin.append(datain[i].split()[use_idex[0]])
                yin.append(datain[i].split()[use_idex[1]])
        else:
            for i in range(len(datain)):
                xin.append(datain[i].split(splitchar)[use_idex[0]])
                yin.append(datain[i].split(splitchar)[use_idex[1]])
    else:
        if splitchar==None:
            for i in range(len(datain)):
                xin[i]= float(datain[i].split()[use_idex[0]])
                yin[i]= float(datain[i].split()[use_idex[1]])
        else:
            for i in range(len(datain)):
                xin[i]= float(datain[i].split(splitchar)[use_idex[0]])
                yin[i]= float(datain[i].split(splitchar)[use_idex[1]])

    return xin,yin

def read_filelist_into_dataframe(file_list, pref_name, junk=5, return_idex_version=False,postname_len=4,backjunk=0):
    """ taking a file_list, remove common 'pref_name' from each file (used for column labels) and return a dataframe with 2-column
    data read accordingly.  junk=length of header data to skip in each file.  backjunk=similar, but end of file.
    postname_len = length of filename to cut off of column-labels (i.e. .dat would be removed if postname_len = 4)"""
    pref_len = len(pref_name)

    q,sq = read_twocol_data(file_list[0],junk=junk,backjunk=backjunk,shh=True)
    df_all_sq = pd.DataFrame(index=q)
    for i in range(len(file_list)):
        if postname_len != 0:
            this_column_name = file_list[i][pref_len:-postname_len]
        else:
            this_column_name = file_list[i][pref_len:]

        q,sq = read_twocol_data(file_list[i],junk=junk,backjunk=backjunk,shh=True)
        df_all_sq[this_column_name] = sq


    return df_all_sq


def batch_process_dfsq_pz_corrections(df_sqfg,bgd,rescale_bgd=1.0,plaz_type=None,
                                     gauss_damp=False,gw=20.0,qmax=None,qmin=None,
                                     rmin=0.0,rmax=20.0,delr=.02
                                     ,qminpla=10.0,qmaxpla=30.0,ndeg=2, return_correction = False,
                                    skip_bgd = False, return_final_sq = False, force_qmax_type='Off'):
    """
    Provide a dataframe (foreground) and a numpy-array background, and batch process the PDFs using the same methods as
    are found in do_reduction_placzek_corrections, returned in a new dataframe.  Note that return_final_sq and
    return_correction WILL work, and you will get a dataframe of the final S(Q)/corrections in Q-space.
    """

    if isinstance(df_sqfg, pd.DataFrame) == False:
        print ("You did not pass me a dataframe.  I am sad now.")
        return None

    else:

        if return_correction == False and return_final_sq == False:

            q = np.array(df_sqfg.index)
            r, first_gr = do_reduction_placzek_corrections(q, np.array(df_sqfg.iloc[:,0]), bgd,
                                                            rescale_bgd=rescale_bgd,skip_bgd =skip_bgd,
                                                            gauss_damp=gauss_damp,gw=gw,
                                                            qmin=qmin,qmax=qmax,
                                                            force_qmax_type=force_qmax_type,
                                                            plaz_type = plaz_type, qminpla=qminpla,qmaxpla=qmaxpla,
                                                            rmax=rmax,delr =delr,rmin=rmin)
            df_all_gr = pd.DataFrame(index=r)
            df_all_gr[df_sqfg.columns[0]] = first_gr

            for i in range(1,len(df_sqfg.columns)):
                col = df_sqfg.columns[i]
                this_sq = np.array(df_sqfg.loc[:,col])
                r, this_gr = do_reduction_placzek_corrections(q, this_sq, bgd,
                                                                rescale_bgd=rescale_bgd,skip_bgd =skip_bgd,
                                                                gauss_damp=gauss_damp,gw=gw,
                                                                qmin=qmin,qmax=qmax,
                                                                force_qmax_type=force_qmax_type,
                                                                plaz_type = plaz_type, qminpla=qminpla,qmaxpla=qmaxpla,
                                                                rmax=rmax,delr =delr,rmin=rmin)
                df_all_gr[col] = this_gr

            return df_all_gr

        else:

            q = np.array(df_sqfg.index)
            sqback = do_reduction_placzek_corrections(q, np.array(df_sqfg.iloc[:,0]), bgd,
                                                            rescale_bgd=rescale_bgd,skip_bgd =skip_bgd,
                                                            gauss_damp=gauss_damp,gw=gw,
                                                            qmin=qmin,qmax=qmax,
                                                            force_qmax_type=force_qmax_type,
                                                            plaz_type = plaz_type, qminpla=qminpla,qmaxpla=qmaxpla,
                                                            rmax=rmax,delr =delr,rmin=rmin,
                                                            return_correction=return_correction,
                                                            return_final_sq= return_final_sq)
            df_back_sq = pd.DataFrame(index=q)
            df_back_sq[df_sqfg.columns[0]] = sqback

            for i in range(1,len(df_sqfg.columns)):
                col = df_sqfg.columns[i]
                this_sq = np.array(df_sqfg.loc[:,col])
                this_sqback = do_reduction_placzek_corrections(q, this_sq, bgd,
                                                                rescale_bgd=rescale_bgd,skip_bgd =skip_bgd,
                                                                gauss_damp=gauss_damp,gw=gw,
                                                                qmin=qmin,qmax=qmax,
                                                                force_qmax_type=force_qmax_type,
                                                                plaz_type = plaz_type, qminpla=qminpla,qmaxpla=qmaxpla,
                                                                rmax=rmax,delr =delr,rmin=rmin,
                                                                return_correction=return_correction,
                                                                return_final_sq= return_final_sq)
                df_back_sq[col] = this_sqback

            return df_back_sq
