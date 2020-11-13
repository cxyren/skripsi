function  cell_array_samples = loadJoints( pathDataset, jointsType )
%LOADJOINTS 
%   load joints from dataset files into a cell_array
%
%cell_array_samples = loadJoints( pathDataset, jointsType )
%
%INPUT: 
%   pathDataset: 
%       directory of KARD dataset
%       default='./'
%   jointsType: 
%       type of coordinate joints, can be 'screen' or 'realworld' 
%       default 'realworld'
%
%OUTPUT:
%   cell_array_samples is a (num_actions, num_execution*num_subjects) array of cells
%   each cell contains a matrix(num_joints, num_frames, num_sizes) 
%
%EXAMPLE:
%   loadJoints('./KARD/', 'realworld')

if nargin == 0
    pathDataset = './';
    jointsType = 'realworld';
    
elseif nargin == 1
    jointsType = 'realworld';
end 

num_actions = 18;
num_subject = 10;
num_executions = 3;
num_sizes = 3;   %(x,y,z)
num_joints = 15;

cell_array_samples = cell(num_actions, num_subject*num_executions);


for action=1%:num_actions
    index_samples = 1;
    for subject=1:num_subject
        
        for execution=1:num_executions
            
            filename = sprintf('a%02i_s%02i_e%02i_',action,subject,execution);
            filename = strcat(pathDataset,filename,jointsType,'.txt')   
            fp=fopen(filename);
            A=fscanf(fp,'%f');
            fclose(fp);          

            l=size(A,1)/num_sizes;
            matrix_joint=reshape(A,num_sizes,l)';
            totElements = size(matrix_joint,1);

            %num_frames = totElements/num_giunti;

            index1 = 1;
            index2 = 1;
            while index1 < totElements

                elencoAttuale = [index1:index1+num_joints-1];

                if (strcmp(jointsType, 'screen') == 1)
                 
                    x = matrix_joint(elencoAttuale,1);       
                    z = matrix_joint(elencoAttuale,3)/10;
                    y = 640-matrix_joint(elencoAttuale,2);  
                    
                elseif (strcmp(jointsType, 'realworld') == 1)
                    x = matrix_joint(elencoAttuale,1);        
                    z = matrix_joint(elencoAttuale,3);
                    y = matrix_joint(elencoAttuale,2);    
                    
                else
                   disp(strcat('Error: ',jointsType, 'must be screen or realworld') ) 
                   
                   
                end
                
                matrix_joints(1:num_joints,index2,:) = [x y z];
                
                
                index1 = index1+num_joints;
                index2 = index2+1;
            end
            
            
            cell_array_samples{action,index_samples} = matrix_joints;
            index_samples = index_samples+1;
            
         end
    end
     
end


end



