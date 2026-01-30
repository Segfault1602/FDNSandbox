function [A, B, C, m] = ReadFDNConfig(filename)
    data = readmatrix(filename);
    B = data(1,:)';
    C = data(2,:);
    
    m = data(3,:);

    A = data(4:end,:);


   
    % gainPerSample = db2mag(RT602slope(10, 48000));
    % absorption = (gainPerSample.^m).';
    % A = A * diag(absorption);
end