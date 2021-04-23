#############################################################
#	Perform Eigen Analysis                              #
#	Authors: Srinivasa B. Ramisetti    	            #
#	Created:   09-September-2020	  	            #
#	E-mail: ramisettisrinivas@yahoo.com		    #
#	Web:	http://ramisetti.github.io		    #
#############################################################
#!/usr/bin/env python

import numpy as np
import scipy.linalg as scp
import mmap
import getMats as gm

def get_tt_inverse(sin_col, cos_col):

    c1 = cos_col[0];
    c2 = cos_col[1];
    c3 = cos_col[2];
    
    s1 = sin_col[0];
    s2 = sin_col[1];
    s3 = sin_col[2];

    
    ttv = [ [c2*s3 - s2*c3,  c3*s1 - s3*c1, c1*s2 - s1*c2],
            [  s2 - s3 ,       s3 - s1,       s1 - s2 ],
            [  c3 - c2 ,       c1 - c3,       c2 - c1 ] ]
    ttv = ttv/(1.5*np.sqrt(3));

    return ttv

def get_new_seq(rot_triplet,ntot):
#  rot_triplet is size n x 3
    #print('rot_triplet ', len(rot_triplet))
    rot_triplet=np.array(rot_triplet,dtype=int)
    if (rot_triplet.size==0):
        nRotTriplets=0
        nb=0
        #return np.array([]),0,0;
    else:
        nRotTriplets,nb = rot_triplet.shape;

    #print(nRotTriplets,nb,rot_triplet.flatten())
    
    if (nb != 3 and nRotTriplets != 0 and ntot!= 0):
        print('**ERROR: the number of column vectors in the rotating triplet must equal 3, the num of blades');
        new_seq = np.range(1,ntot)
    else:
        non_rotating = np.ones(ntot,dtype=int);
        #print(non_rotating)
        non_rotating[rot_triplet.flatten()] = 0; # if they are rotating, set them false;
        a=np.array(np.nonzero(non_rotating)).flatten()
        b=(rot_triplet.reshape(nRotTriplets*nb, 1)).flatten()
        new_seq = np.concatenate((a,b));

        #print(new_seq)
    return new_seq,nRotTriplets,nb

def eiganalysis(A, ndof2, ndof1):
    # this line has to be the first line for this function
    # to get the number of function arguments
    nargin=len(locals())

    mbc={}
    m, ns = A.shape;
    #print(ndof2,ndof1, m,ns)
    if(m!=ns):
        sys.exit('**ERROR: the state-space matrix is not a square matrix.');

    if nargin == 1:
        ndof1 = 0;
        ndof2 = ns/2;

        if np.mod(ns,2) != 0:
            print('**ERROR: the input matrix is not of even order.');
    elif nargin == 2:
        ndof1 = ns - 2*ndof2;
        if ndof1 < 0:
            print('**ERROR: ndof2 must be no larger than half the dimension of the state-space matrix.');      
    else:
        if ns != 2*ndof2 + ndof1:
            print('**ERROR: the dimension of the state-space matrix must equal 2*ndof2 + ndof1.');

    ndof = ndof2 + ndof1;

    #print(ndof, ndof2,ndof1, m,ns)
    origEvals, origEigenVects = np.linalg.eig(A); #,'nobalance'
    # errorInSolution = norm(A * mbc.EigenVects - mbc.EigenVects* diag(mbc.EigenVals) )
    # these eigenvalues aren't sorted, so we just take the ones with
    # positive imaginary parts to get the pairs for modes with damping < 1:
    positiveImagEvals = np.argwhere( origEvals.imag > 0.0);
    
    mbc['Evals']             = origEvals[positiveImagEvals];
    row=np.array(list(range(0,ndof2))+list(range(ndof2*2+1,ns)))
    col=positiveImagEvals

    mbc['EigenVects']=origEigenVects[row,col].transpose(); # save q2 and q1, throw away q2_dot
    # print('GGGGGG ', row.shape,col.shape)
    # with open('AAAA', "a") as f:
    #     np.savetxt(f,mbc['EigenVects'].imag,fmt='%5.4f')
    #     f.write('\n')

    EigenVects_save=origEigenVects[:,positiveImagEvals]; # save these for VTK visualization;

    #print('EigenVects_save shape ',EigenVects_save[:,:,0].shape, origEigenVects.shape)

    real_Evals = mbc['Evals'].real;
    imag_Evals = mbc['Evals'].imag;

    mbc['NaturalFrequencies'] = np.sqrt( real_Evals**2 + imag_Evals**2 );
    mbc['DampRatios'] = -real_Evals/mbc['NaturalFrequencies']
    mbc['DampedFrequencies']  = imag_Evals;

    mbc['NumRigidBodyModes'] = ndof - len(positiveImagEvals);

    mbc['NaturalFreqs_Hz'] = mbc['NaturalFrequencies']/(2.0*np.pi)
    mbc['DampedFreqs_Hz']  = mbc['DampedFrequencies']/(2.0*np.pi);
    mbc['MagnitudeModes']  = np.abs(mbc['EigenVects']);
    mbc['PhaseModes_deg']  = np.angle(mbc['EigenVects'])*180.0/np.pi;

    # print(real_Evals[0:10])
    # print(imag_Evals[0:10])
    # print(mbc['NaturalFrequencies'][0:10])
    # print(mbc['DampRatios'][0:10])
    # print(mbc['NaturalFreqs_Hz'][0:10])
    # print(mbc['DampedFreqs_Hz'][0:10])
    # print(mbc['MagnitudeModes'].shape)
    # print(mbc['PhaseModes_deg'][0,0])

    # with open('MagModes.txt', "a") as f:
    #     np.savetxt(f,mbc['MagnitudeModes'],fmt='%g')
    #     f.write('\n')

    # with open('PhsModes.txt', "a") as f:
    #     np.savetxt(f,mbc['PhaseModes_deg'],fmt='%g')
    #     f.write('\n')

    # with open('NFreqs_Hz.txt', "a") as f:
    #     np.savetxt(f,mbc['NaturalFreqs_Hz'],fmt='%g')
    #     f.write('\n')
    # with open('DampRatios.txt', "a") as f:
    #     np.savetxt(f,mbc['DampRatios'],fmt='%g')
    #     f.write('\n')
    # with open('DFreqs_Hz.txt', "a") as f:
    #     np.savetxt(f,mbc['DampedFreqs_Hz'],fmt='%g')
    #     f.write('\n')        

    return mbc,EigenVects_save[:,:,0]

def fx_mbc3(FileNames):
    MBC={}
    matData, FAST_linData = gm.get_Mats(FileNames)

    # print('matData[Omega] ', matData['Omega'])
    # print('matData[OmegaDot] ', matData['OmegaDot'])
    
    MBC['DescStates'] = matData['DescStates'] # save this in the MBC type for possible campbell_diagram processing later 
    MBC['ndof2'] = matData['ndof2']
    MBC['ndof1'] = matData['ndof1']
    MBC['RotSpeed_rpm'] = np.mean(matData['Omega'])*(30/np.pi); #rad/s to rpm
    
    if 'WindSpeed' in matData:
        MBC['WindSpeed'] = np.mean(matData['WindSpeed'])
        
    # print('RotSpeed_rpm ',MBC['RotSpeed_rpm'])
    # print('ndof1 ', MBC['ndof1'])
    # print('ndof2 ', MBC['ndof2'])
    # print(matData['RotTripletIndicesStates2'])
    
    #  nb = 3; % number of blades required for MBC3
    # ---------- Multi-Blade-Coordinate transformation -------------------------------------------
    new_seq_dof2, dummy, nb = get_new_seq(matData['RotTripletIndicesStates2'],matData['ndof2']); # these are the first ndof2 states (not "first time derivative" states); these values are used to calculate matrix transformations
    new_seq_dof1, dummy, dummy = get_new_seq(matData['RotTripletIndicesStates1'],matData['ndof1']); # these are the first-order ndof1 states; these values are used to calculate matrix transformations

    # print('new_seq_dof2 ', new_seq_dof2)
    # print('new_seq_dof1 ', new_seq_dof1)
    # print('dummy ', dummy, ' nb ', nb)

    new_seq_states=np.concatenate((new_seq_dof2, new_seq_dof2+matData['ndof2']))
    if new_seq_dof1.size!=0:
        new_seq_states=np.concatenate((new_seq_states,new_seq_dof1+matData['NumStates2']))
    
    #new_seq_states = [new_seq_dof2;  new_seq_dof2+matData['ndof2'];  new_seq_dof1+matData['NumStates2']]; # combine the second-order states, including "first time derivatives", with first-order states (assumes ordering of displacements and velocities in state matrices); these values are used to calculate matrix transformations 
    # second-order NonRotating q2, second-order Rotating q2, 
    # second-order NonRotating q2_dot, second-order Rotating q2_dot, 
    # first-order NonRotating q1, first-order Rotating q1


    if nb == 3:
        MBC['performedTransformation'] = True;

        if matData['n_RotTripletStates2'] + matData['n_RotTripletStates1'] < 1:
            print('*** There are no rotating states. MBC transformation, therefore, cannot be performed.');
        # perhaps just warn and perform eigenanalysis anyway?
        elif (matData['n_RotTripletStates2']*nb > matData['ndof2']):
            print('**ERROR: the rotating second-order dof exceeds the total num of second-order dof');
        elif (matData['n_RotTripletStates1']*nb > matData['ndof1']):
            print('**ERROR: the rotating first-order dof exceeds the total num of first-order dof');

        new_seq_inp,dummy,dummy = get_new_seq(matData['RotTripletIndicesCntrlInpt'],matData['NumInputs']);
        new_seq_out,dummy,dummy = get_new_seq(matData['RotTripletIndicesOutput'],matData['NumOutputs']);

        n_FixFrameStates2 = matData['ndof2']      - matData['n_RotTripletStates2']*nb;  # fixed-frame second-order dof
        n_FixFrameStates1 = matData['ndof1']      - matData['n_RotTripletStates1']*nb;  # fixed-frame first-order dof
        n_FixFrameInputs  = matData['NumInputs']  - matData['n_RotTripletInputs']*nb;   # fixed-frame control inputs
        n_FixFrameOutputs = matData['NumOutputs'] - matData['n_RotTripletOutputs']*nb;  # fixed-frame outputs

        #print(n_FixFrameOutputs,n_FixFrameInputs, n_FixFrameStates1, n_FixFrameStates2)

        if ( len(matData['Omega']) != matData['NAzimStep']):
            print('**ERROR: the size of Omega vector must equal matData.NAzimStep, the num of azimuth steps')
        if ( len(matData['OmegaDot']) != matData['NAzimStep']):
            print('**ERROR: the size of OmegaDot vector must equal matData.NAzimStep, the num of azimuth steps');


        MBC['A']=np.zeros(matData['A'].shape)
        MBC['B']=np.zeros((len(new_seq_states),len(new_seq_inp),matData['NAzimStep']))
        MBC['C']=np.zeros(matData['C'].shape)
        MBC['D']=np.zeros(matData['D'].shape)

        # print('new_seq_inp ',new_seq_inp)
        # print('new_seq_out ',new_seq_out)
        # print('new_seq_states ', new_seq_states)
            
        # begin azimuth loop 
        for iaz in reversed(range(matData['NAzimStep'])):
            #(loop backwards so we don't reallocate memory each time [i.e. variables with iaz index aren't getting larger each time])

            temp=np.arange(nb)
            # compute azimuth positions of blades:
            az = matData['Azimuth'][iaz]*np.pi/180.0 + 2*np.pi/nb* temp ; # Eq. 1, azimuth in radians

            # get rotor speed squared
            OmegaSquared = matData['Omega'][iaz]**2;

            #print(OmegaSquared)

            # compute transformation matrices
            cos_col = np.cos(az);
            sin_col = np.sin(az);

            tt=np.column_stack((np.ones(3),cos_col,sin_col))  # Eq. 9, t_tilde
            ttv = get_tt_inverse(sin_col, cos_col);     # inverse of tt (computed analytically in function below)
            tt2 = np.column_stack((np.zeros(3), -sin_col,  cos_col))     # Eq. 16 a, t_tilde_2
            tt3 = np.column_stack((np.zeros(3), -cos_col, -sin_col))     # Eq. 16 b, t_tilde_3
            
            #---
            T1 = np.eye(n_FixFrameStates2);                # Eq. 11 for second-order states only
            #print('B ',T1, n_FixFrameStates2, matData['n_RotTripletStates2'])
            for ii in range(matData['n_RotTripletStates2']):
                T1 = scp.block_diag(T1,tt)
            
            T1v = np.eye(n_FixFrameStates2);               # inverse of T1
            for ii in  range(matData['n_RotTripletStates2']):
                T1v = scp.block_diag(T1v, ttv);

            T2 = np.zeros([n_FixFrameStates2,n_FixFrameStates2]);              # Eq. 14  for second-order states only
            for ii in range(matData['n_RotTripletStates2']):
                T2 = scp.block_diag(T2, tt2);

            #print('T1, T1v, T2 ',T1.shape, T1v.shape, T2.shape)
            #---    
            T1q = np.eye(n_FixFrameStates1);               # Eq. 11 for first-order states (eq. 8 in MBC3 Update document)
            for ii in range(matData['n_RotTripletStates1']):
                T1q = blkdiag(T1, tt);

            T1qv = np.eye(n_FixFrameStates1);              # inverse of T1q
            for ii in range(matData['n_RotTripletStates1']):
                T1qv = blkdiag(T1qv, ttv);

            T2q = np.zeros([n_FixFrameStates1,n_FixFrameStates1]);             # Eq. 14 for first-order states (eq.  9 in MBC3 Update document)
            for ii in range(matData['n_RotTripletStates1']):
                T2q = blkdiag(T2q, tt2);

            #print('T1q, T1qv, T2q ',T1q.shape, T1qv.shape, T2q.shape)
            #     T1qc = np.eye(matData.NumHDInputs);            # inverse of T1q

            #---
            T3 = np.zeros([n_FixFrameStates2,n_FixFrameStates2]);              # Eq. 15
            for ii in range(matData['n_RotTripletStates2']):
                T3 = scp.block_diag(T3, tt3);

            #---
            T1c = np.eye(n_FixFrameInputs);                # Eq. 21
            for ii in range(matData['n_RotTripletInputs']):
                T1c = scp.block_diag(T1c, tt)

            T1ov = np.eye(n_FixFrameOutputs);              # inverse of Tlo (Eq. 23)
            for ii in range(matData['n_RotTripletOutputs']):
                T1ov = scp.block_diag(T1ov, ttv);

            #print('T3, T1c, T1ov ',T3.shape, T1c.shape, T1ov.shape, matData['A'].shape)
            # mbc transformation of first-order matrices
            #  if ( MBC.EqnsOrder == 1 ) # activate later

            #print('Before ',T1c)
    
            if 'A' in matData:
                # Eq. 29
                L1=np.concatenate((T1, np.zeros([matData['ndof2'],matData['ndof2']]), np.zeros([matData['ndof2'], matData['ndof1']])), axis=1)
                L2=np.concatenate((matData['Omega'][iaz]*T2,T1,np.zeros([matData['ndof2'], matData['ndof1']])), axis=1)
                L3=np.concatenate((np.zeros([matData['ndof1'], matData['ndof2']]),np.zeros([matData['ndof1'], matData['ndof2']]), T1q), axis=1)            
                L=np.matmul(matData['A'][new_seq_states[:,None],new_seq_states,iaz],np.concatenate((L1,L2,L3),axis=0))

                R1=np.concatenate((matData['Omega'][iaz]*T2, np.zeros([matData['ndof2'],matData['ndof2']]), np.zeros([matData['ndof2'], matData['ndof1']])), axis=1)
                R2=np.concatenate((OmegaSquared*T3 + matData['OmegaDot'][iaz]*T2,  2*matData['Omega'][iaz]*T2, np.zeros([matData['ndof2'], matData['ndof1']])),axis=1)
                R3=np.concatenate((np.zeros([matData['ndof1'], matData['ndof2']]), np.zeros([matData['ndof1'], matData['ndof2']]),  matData['Omega'][iaz]*T2q), axis=1)
        
                R=np.concatenate((R1,R2,R3),axis=0)

                MBC['A'][new_seq_states[:,None],new_seq_states,iaz]=np.matmul(scp.block_diag(T1v, T1v, T1qv),(L-R))

                ffname='AAA'+str(iaz)+'.txt'
                with open(ffname, "a") as f:
                    np.savetxt(f,MBC['A'][:,:,iaz],fmt='%5.4f')
                    f.write('\n')

            if 'B' in matData:
                # Eq. 30
                MBC['B'][new_seq_states[:,None],new_seq_inp,iaz]=np.matmul(np.matmul(scp.block_diag(T1v, T1v, T1qv), matData['B'][new_seq_states[:,None],new_seq_inp,iaz]),T1c)
            
                # ffname='BBB'+str(iaz)+'.txt'
                # with open(ffname, "a") as f:
                #     np.savetxt(f,MBC['B'][:,:,iaz],fmt='%5.4f')
                #     f.write('\n')

            if 'C' in matData:
                # Eq. 31

                L1=np.concatenate((T1, np.zeros([matData['ndof2'],matData['ndof2']]), np.zeros([matData['ndof2'], matData['ndof1']])),axis=1)
                L2=np.concatenate((matData['Omega'][iaz]*T2, T1, np.zeros([matData['ndof2'], matData['ndof1']])), axis=1)
                L3=np.concatenate((np.zeros([matData['ndof1'], matData['ndof2']]), np.zeros([matData['ndof1'], matData['ndof2']]), T1q), axis=1)
            
                MBC['C'][new_seq_out[:,None], new_seq_states,iaz]=np.matmul(np.matmul(T1ov,matData['C'][new_seq_out[:,None],new_seq_states,iaz]),np.concatenate((L1,L2,L3),axis=0))

                # ffname='CCC'+str(iaz)+'.txt'
                # with open(ffname, "a") as f:
                #     np.savetxt(f,MBC['C'][:,:,iaz],fmt='%5.4f')
                #     f.write('\n')

            if 'D' in matData:
               # Eq. 32
                MBC['D'][new_seq_out[:,None],new_seq_inp,iaz] = np.matmul(np.matmul(T1ov,matData['D'][new_seq_out[:,None],new_seq_inp,iaz]), T1c)

                # ffname='DDD'+str(iaz)+'.txt'
                # with open(ffname, "a") as f:
                #     np.savetxt(f,MBC['D'][:,:,iaz],fmt='%5.4f')
                #     f.write('\n')

        # end   # end of azimuth loop
    else:
        print(' fx_mbc3 WARNING: Number of blades is ', str(nb), ' not 3. MBC transformation was not performed.')
        MBC['performedTransformation'] = False;
    
        # initialize matrices
        if 'A' in matData:
            MBC['A'] = matData['A'] # initalize matrix
        if 'B' in matData:
            MBC['B'] = matData['B'] # initalize matrix
        if 'C' in matData:
            MBC['C'] = matData['C'] # initalize matrix
        if 'D' in matData:
            MBC['D'] = matData['D'] # initalize matrix

    # ------------- Eigensolution and Azimuth Averages -------------------------
    if 'A' in MBC:
        MBC['AvgA'] = np.mean(MBC['A'],axis=2); # azimuth-average of azimuth-dependent MBC.A matrices
        MBC['eigSol'], EigenVects_save = eiganalysis(MBC['AvgA'],matData['ndof2'], matData['ndof1']);

        # ffname='AAA_avg'+'.txt'
        # with open(ffname, "a") as f:
        #     np.savetxt(f,MBC['AvgA'],fmt='%5.4f')
        #     f.write('\n')

    # save eigenvectors (doing inverse of MBC3) for VTK visualization in FAST
    # if nargout > 3 or nargin > 1:
    #     [VTK] = GetDataForVTK(MBC, matData, nb, EigenVects_save);
    #     if nargin > 1
    #         WriteDataForVTK(VTK, ModeVizFileName)

    if 'B' in MBC:
        MBC['AvgB'] = np.mean(MBC['B'],axis=2); # azimuth-average of azimuth-dependent MBC.B matrices
        # ffname='BBB_avg'+'.txt'
        # with open(ffname, "a") as f:
        #     np.savetxt(f,MBC['AvgB'],fmt='%5.4f')
        #     f.write('\n')

    if 'C' in MBC:
        MBC['AvgC'] = np.mean(MBC['C'],axis=2); # azimuth-average of azimuth-dependent MBC.C matrices
        # ffname='CCC_avg'+'.txt'
        # with open(ffname, "a") as f:
        #     np.savetxt(f,MBC['AvgC'],fmt='%5.4f')
        #     f.write('\n')

    if 'D' in MBC:
        MBC['AvgD'] = np.mean(MBC['D'],axis=2); # azimuth-average of azimuth-dependent MBC.D matrices
        # ffname='DDD_avg'+'.txt'
        # with open(ffname, "a") as f:
        #     np.savetxt(f,MBC['AvgD'],fmt='%5.4f')
        #     f.write('\n')

    return MBC, matData, FAST_linData
