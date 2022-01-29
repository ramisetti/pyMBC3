#############################################################
#	Perform MBC3 Analysis and plot Campbell Diagram     #
#	Authors: Srinivasa B. Ramisetti    	            #
#	Created:   18-July-2020		  	            #
#	Revised:   11-September-2020		            #
#	E-mail: ramisettisrinivas@yahoo.com		    #
#	Web:	http://ramisetti.github.io		    #
#############################################################
#!/usr/bin/env python

import re, os, sys
import numpy as np
import pandas as pd
import plotCampbellData as pCD
import eigenanalysis as eigAnl
import postMBC as pMBC

def applyMBC(FileNames,NLinTimes=None):
    CampbellData={}
    MBCData=[]
    HubRad=None;TipRad=None;
    BladeLen=None; TowerHt=None
    dataFound=False;
    indx=0
    
    for i in range(len(FileNames)):
        basename=os.path.splitext(os.path.basename(FileNames[i]))[0]
        with open(FileNames[i]) as f:
            datafile = f.readlines()

        if dataFound==False:
            for line in datafile:
                if 'EDFile' in line:
                    EDFile=line.split()[0].replace('"','')
                    with open(EDFile) as edfile:
                        for edline in edfile:
                            if 'HubRad' in  edline:
                                HubRad=float(edline.split()[0])
                            elif 'TipRad' in  edline:
                                TipRad=float(edline.split()[0])
                            elif 'TowerHt' in  edline:
                                TowerHt=float(edline.split()[0])
                            
        if((TipRad!=None and HubRad!=None) or TowerHt!=None):
            BladeLen=TipRad-HubRad
            dataFound=True
        if(TowerHt==None or BladeLen==None):
            print('TowerHt and BladeLen are not available!');
            sys.exit()

        #print('Tower ht ', TowerHt, 'Hub radius', HubRad, 'Tip radius ', TipRad, 'Blade length ', BladeLen)
        found=False
        for line in datafile:
            if found==False and 'NLinTimes' in line:
                NLinTimes=int(line.split()[0])
                found=True
                
        if NLinTimes<1:
            print('NLinTimes should be greater than 0!')
            sys.exit()
            
        linFileNames=[basename+'.'+format(x, 'd')+'.lin' for x in range(1,NLinTimes+1)]

        linFileFlag=True;
        for f in linFileNames:
            if(os.path.isfile(f)!=True):
                linFileFlag=False

        if(linFileFlag):
            print('Processing ', FileNames[i], ' file!', '   number of Linearization files to process ', NLinTimes)
            MBC_data,getMatData,FAST_linData=eigAnl.fx_mbc3(linFileNames)
            MBCData.append(MBC_data)
            print('Multi-Blade Coordinate transformation completed!');
            print('  ');
            CampbellData[indx]=campbell_diagram_data(MBC_data,BladeLen,TowerHt)
            indx=indx+1;

    return CampbellData, MBCData, BladeLen, TowerHt

def campbell_diagram_data(mbc_data, BladeLen, TowerLen):
    CampbellData={}
    usePercent = True;
    #
    # mbc_data.eigSol = eiganalysis(mbc_data.AvgA);
    ndof = mbc_data['ndof2'] + mbc_data['ndof1']; #size(mbc_data.AvgA,1)/2;          # number of translational states
    nModes = len(mbc_data['eigSol']['Evals'])
    #print(nModes)
    DescStates = PrettyStateDescriptions(mbc_data['DescStates'], mbc_data['ndof2'], mbc_data['performedTransformation']);

    ## store indices of max mode for state and to order natural frequencies
    #StatesMaxMode_vals = np.amax(mbc_data['eigSol']['MagnitudeModes'],axis=1); # find which mode has the maximum value for each state (max of each row before scaling)
    StatesMaxMode = np.argmax(mbc_data['eigSol']['MagnitudeModes'],axis=1); # find which mode has the maximum value for each state (max of each row before scaling)
    SortedFreqIndx = np.argsort((mbc_data['eigSol']['NaturalFreqs_Hz']).flatten(),kind="heapsort");
    #print(SortedFreqIndx)


    if BladeLen!=0 or TowerLen!=0:
        ## get the scaling factors for the mode rows
        ScalingFactor = pMBC.getScaleFactors(DescStates, TowerLen, BladeLen);

        ## scale the magnitude of the modes by ScalingFactor (for consistent units)
        #  and then scale the columns so that their maximum is 1

        ModesMagnitude = np.matmul(np.diag(ScalingFactor), mbc_data['eigSol']['MagnitudeModes']); # scale the rows
        #print(ModesMagnitude)
        
        CampbellData['ScalingFactor'] = ScalingFactor;
    else:
        ModesMagnitude = mbc_data['eigSol']['MagnitudeModes'];

    if usePercent:
        scaleCol = 0.01*np.sum( ModesMagnitude,axis=0); # find the sum of the column, and multiply by 100 (divide here) to get a percentage
    else:
        scaleCol = np.amax(ModesMagnitude,axis=0); #find the maximum value in the column, so the first element has value of 1

    ModesMagnitude = np.matmul(ModesMagnitude,np.diag(1./scaleCol)) # scale the columns

    CampbellData['NaturalFreq_Hz'] = mbc_data['eigSol']['NaturalFreqs_Hz'][SortedFreqIndx]
    CampbellData['DampingRatio']   = mbc_data['eigSol']['DampRatios'][SortedFreqIndx]
    CampbellData['RotSpeed_rpm']   = mbc_data['RotSpeed_rpm']

    CampbellData['DampedFreq_Hz'] = mbc_data['eigSol']['DampedFreqs_Hz'][SortedFreqIndx]
    CampbellData['MagnitudePhase'] =  mbc_data['eigSol']['MagnitudeModes']
    CampbellData['PhaseDiff'] =  mbc_data['eigSol']['PhaseModes_deg']

    if 'WindSpeed' in mbc_data:
        CampbellData['WindSpeed']  = mbc_data['WindSpeed']

    #print(ModesMagnitude.shape)
    CampbellData['Modes']=[]

    for i in range(nModes):
        CData={}
        CData['NaturalFreq_Hz'] = mbc_data['eigSol']['NaturalFreqs_Hz'][SortedFreqIndx[i]]
        CData['DampedFreq_Hz']  = mbc_data['eigSol']['DampedFreqs_Hz'][SortedFreqIndx[i]];
        CData['DampingRatio']   = mbc_data['eigSol']['DampRatios'][SortedFreqIndx[i]];

        
        #print(np.argsort(ModesMagnitude[:,SortedFreqIndx[0]])[::-1])
        # sort indices in descending order
        sort_state = np.argsort( ModesMagnitude[:,SortedFreqIndx[i]])[::-1];

        #print(type(sort_state))
        CData['DescStates']=[DescStates[i] for i in sort_state]
        CData['MagnitudePhase']=ModesMagnitude[sort_state,SortedFreqIndx[i]];
        Phase =                mbc_data['eigSol']['PhaseModes_deg'][sort_state,SortedFreqIndx[i]];
        # if the phase is more than +/- 90 degrees different than the first
        # one (whose value == 1 or is the largest %), we'll stick a negative value on the magnitude:
        Phase = np.mod(Phase, 360);
    
        CData['PhaseDiff'] = np.mod( Phase - Phase[0], 360); # difference in range [0, 360)
        PhaseIndx = CData['PhaseDiff'] > 180;
        CData['PhaseDiff'][PhaseIndx] = CData['PhaseDiff'][PhaseIndx] - 360;   # move to range (-180, 180]
    
        if ~usePercent:
            PhaseIndx = CData['PhaseDiff'] > 90;
            CData['MagnitudePhase'][PhaseIndx] = -1*CData['MagnitudePhase'][PhaseIndx];
            CData['PhaseDiff'][PhaseIndx] = CData['PhaseDiff'][PhaseIndx] - 180;

            PhaseIndx = CData['PhaseDiff'] <= -90;
            CData['MagnitudePhase'][PhaseIndx] = -1*CData['MagnitudePhase'][PhaseIndx];
            CData['PhaseDiff'][PhaseIndx] = CData['PhaseDiff'][PhaseIndx] + 180;

        #print(CData['MagnitudePhase'])
        #print(CData['PhaseDiff'])

        CData['StateHasMaxAtThisMode'] = np.ones(ndof, dtype=bool);
        ix = (StatesMaxMode == SortedFreqIndx[i]);
        tmp=ix[sort_state]
        CData['StateHasMaxAtThisMode']=tmp
                    
        #print(CData['StateHasMaxAtThisMode'])
        #print(CData['NaturalFreq_Hz'])
        CampbellData['Modes'].append(CData)

    #print(CampbellData[0]['MagnitudePhase'])

    CampbellData['nColsPerMode'] = 5;
    CampbellData['ModesTable'] = {}

    for i in range(nModes):
        colStart = i*CampbellData['nColsPerMode'];
        CampbellData['ModesTable'][1, colStart+1 ] = 'Mode number:';
        CampbellData['ModesTable'][1, colStart+2 ] = i;

        CampbellData['ModesTable'][2, colStart+1 ] = 'Natural (undamped) frequency (Hz):';
        CampbellData['ModesTable'][2, colStart+2 ] = np.asscalar(CampbellData['Modes'][i]['NaturalFreq_Hz'])

        CampbellData['ModesTable'][3, colStart+1 ] = 'Damped frequency (Hz):';
        CampbellData['ModesTable'][3, colStart+2 ] = np.asscalar(CampbellData['Modes'][i]['DampedFreq_Hz'])

        CampbellData['ModesTable'][4, colStart+1 ] = 'Damping ratio (-):';
        CampbellData['ModesTable'][4, colStart+2 ] = np.asscalar(CampbellData['Modes'][i]['DampingRatio'])
        
        CampbellData['ModesTable'][5, colStart+1 ] = 'Mode ' + str(i) + ' state description';
        CampbellData['ModesTable'][5, colStart+2 ] = 'State has max at mode ' + str(i);
        if usePercent:
            CampbellData['ModesTable'][5, colStart+3 ] = 'Mode ' + str(i) + ' contribution (%)';
        else:
            CampbellData['ModesTable'][5, colStart+3 ] = 'Mode ' + str(i) + ' signed magnitude';

        CampbellData['ModesTable'][5, colStart+4 ] = 'Mode ' + str(i) + ' phase (deg)';

        # need to cross check these 4 lines
        CampbellData['ModesTable'][6,colStart+1] = CampbellData['Modes'][i]['DescStates'];
        CampbellData['ModesTable'][6,colStart+2] = CampbellData['Modes'][i]['StateHasMaxAtThisMode'];
        CampbellData['ModesTable'][6,colStart+3] = CampbellData['Modes'][i]['MagnitudePhase'];
        CampbellData['ModesTable'][6,colStart+4] = CampbellData['Modes'][i]['PhaseDiff'];
        #print(CampbellData['ModesTable'][6,colStart+3])

    pd.DataFrame(CampbellData['ModesTable']).to_csv('myfile.csv', header=False, index=False)
    # with open('dict.csv', 'w') as csv_file:  
    #     writer = csv.writer(csv_file)
    #     for key, value in CampbellData['ModesTable'].items():
    #         writer.writerow([key, value])
    return CampbellData
    

def PrettyStateDescriptions(DescStates, ndof2, performedTransformation):
    idx=np.array(list(range(0,ndof2))+list(range(ndof2*2+1,len(DescStates))))    
    tmpDS = [DescStates[i] for i in idx]
    
    if performedTransformation:
        key_vals=[['BD_1','Blade collective'],['BD_2','Blade cosine'],['BD_3','Blade sine'],['blade 1','blade collective'], ['blade 2','blade cosine'], ['blade 3','Blade sine '], ['PitchBearing1','Pitch bearing collective '], ['PitchBearing2','Pitch bearing cosine '], ['PitchBearing3','Pitch bearing sine ']]
        # Replace Substrings from String List 
        sub = dict(key_vals)
        for key, val in sub.items(): 
            for idx, ele in enumerate(tmpDS):
                if key in ele: 
                    tmpDS[idx] = ele.replace(key, val)

        StateDesc=tmpDS
    else:
        StateDesc = tmpDS
    
    for i in range(len( StateDesc )):
        First=re.split('\(',StateDesc[i],2)
        Last=re.split('\)',StateDesc[i],2)

        if len(First)>0 and len(Last)>0 and len(First[0]) != len(StateDesc[i]) and len( Last[-1] ) != len(StateDesc[i]):
            StateDesc[i] = (First[0]).strip() + Last[-1];
            #print(StateDesc[i])
        
    return StateDesc

def main():
    # read wind/rotor speeds and fast linearization files from input file
    FileNames,OP,PltForm=pMBC.getFastFiles(sys.argv[1])
    CampbellData,MBC_Data,BladeLen,TowerHt=applyMBC(FileNames)
    print('Preparing campbell diagram data!');

    if not os.path.exists('./results'):
        os.mkdir('./results')

    #uncomment the below line to write the v2 xlsm file
    #pMBC.writeExcelData(MBC_Data,OP,BladeLen,TowerHt,PltForm);

    modeID_table,modesDesc=pMBC.IdentifyModes(CampbellData)

    print(modeID_table,len(OP))
    nModes=modeID_table.shape[0]
    nRuns=modeID_table.shape[1]
    cols=[item[0] for item in list(modesDesc.values())]

    frequency=pd.DataFrame(np.nan, index=np.arange(nRuns), columns=cols)
    dampratio=pd.DataFrame(np.nan, index=np.arange(nRuns), columns=cols)
    print(nRuns,nModes)

    FreqPlotData=np.zeros((nRuns,nModes))
    DampPlotData=np.zeros((nRuns,nModes))
    for i in range(nRuns):
        for modeID in range(len(modesDesc)): # list of modes we want to identify
            idx=int(modeID_table[modeID,i])
            FreqPlotData[i,modeID]=CampbellData[i]['Modes'][idx]['NaturalFreq_Hz']
            DampPlotData[i,modeID]=CampbellData[i]['Modes'][idx]['DampingRatio']
            #print(i,modeID,modesDesc[modeID][0],FreqPlotData[i,modeID])
        frequency.iloc[i,:]=FreqPlotData[i,:]
        dampratio.iloc[i,:]=DampPlotData[i,:]

    for i in range(len(OP)):
        # for 15 DOFs
        frequency.index.values[i]=OP[i]
        dampratio.index.values[i]=OP[i]

    # drop columns not required
    frequency=frequency.drop(['Generator DOF (not shown)', 'Nacelle Yaw (not shown)'], axis = 1)
    #frequency.drop(['Generator DOF (not shown)'], axis = 1) 
    dampratio=dampratio.drop(['Generator DOF (not shown)', 'Nacelle Yaw (not shown)'], axis = 1)

    pCD.plotCampbellData(OP,frequency,dampratio)

    lenColumns=len(frequency.columns)

    frequency['1P']=np.nan
    frequency['3P']=np.nan
    frequency['6P']=np.nan
    frequency['9P']=np.nan
    frequency['12P']=np.nan

    for i in range(nRuns):
        # for 1P,3P,6P,9P,and 12P harmonics
        tmp=OP[i]/60.0
        frequency.iloc[i,lenColumns]=tmp
        frequency.iloc[i,lenColumns+1]=3*tmp
        frequency.iloc[i,lenColumns+2]=6*tmp
        frequency.iloc[i,lenColumns+3]=9*tmp
        frequency.iloc[i,lenColumns+4]=12*tmp

    # uncomment to write excel file with transposed frequency data
    #frequency.transpose().to_excel(r'CampbellData.xlsx')

    # dbgwriter = pd.ExcelWriter('CampbellTable.xlsx', engine='xlsxwriter')
    # dfggg = pd.DataFrame(CampbellData[0]['ModesTable'])
    # dfggg.to_excel(dbgwriter)
    # dbgwriter.save()

    writer = pd.ExcelWriter('./results/CampbellData.xlsx', engine='xlsxwriter')
    frequency.to_excel(writer,sheet_name='FrequencyHz')
    dampratio.to_excel(writer,sheet_name='DampingRatios')

    #writer.save()
    #exit()
    # for debugging purpose
    maxsize=0
    for indx in range(len(CampbellData)):
        tmp=CampbellData[indx]['NaturalFreq_Hz'].shape[0]
        #print('Shape ', CampbellData[indx]['NaturalFreq_Hz'])
        if (maxsize<tmp):
            maxsize=tmp

    print('Debug Info: max number of DOFs ', maxsize)
    print('Len CampbellData', len(CampbellData))
    #tmpFreq=np.empty([len(CampbellData),maxsize])
    #tmpDamp=np.empty([len(CampbellData),maxsize])
    tmpFreq=pd.DataFrame(np.nan, index=np.arange(len(CampbellData)),columns=np.arange(maxsize))
    tmpDamp=pd.DataFrame(np.nan, index=np.arange(len(CampbellData)),columns=np.arange(maxsize))
    #print(len(cols),tmpFreq.shape,CampbellData[0]['NaturalFreq_Hz'].shape)
    for indx in range(len(CampbellData)):
        addsize=(maxsize-CampbellData[indx]['NaturalFreq_Hz'].shape[0])
        a=CampbellData[indx]['NaturalFreq_Hz']
        tmpArr=1E-10*np.ones(addsize)
        #print(addsize,tmpFreq.shape, a.shape)
        a=np.append(tmpArr,a)
        tmpFreq.iloc[indx,:]=a

    addsize=(maxsize-CampbellData[indx]['DampingRatio'].shape[0])
    tmpArr=1E-10*np.ones(addsize)
    b=CampbellData[indx]['DampingRatio']
    b=np.append(tmpArr,b)
    tmpDamp.iloc[indx,:]=b

    tmpFreq.to_excel(writer,sheet_name='SortedFreqHz')
    tmpDamp.to_excel(writer,sheet_name='SortedDampRatios')

    # tmpFreqFile="FreqHz.txt"
    # tmpDampFile="DampingRatios.txt"
    # with open(tmpFreqFile, "a") as f:
    #     np.savetxt(f,tmpFreq,fmt='%g', delimiter=' ', newline=os.linesep)
    
    # with open(tmpDampFile, "a") as f:
    #     np.savetxt(f,tmpDamp,fmt='%g', delimiter=' ', newline=os.linesep)
    
    # print(CampbellData[indx]['NaturalFreq_Hz'])
    # print(CampbellData[indx]['DampingRatio'])
    # end of debugging

    # save to excel file and close
    writer.save()

if __name__ == "__main__":
    main()

#End of script
