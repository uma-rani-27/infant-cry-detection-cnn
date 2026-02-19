function feat = extract_features(signal, fs)
% -------------------------------------------------------------
% Function: extract_features
% Purpose : Extract MFCC features from infant cry audio signal
% Input   : signal - audio signal
%           fs     - sampling frequency
% Output  : feat   - extracted feature vector
% -------------------------------------------------------------

    % Convert stereo audio to mono (if needed)
    if size(signal, 2) > 1
        signal = mean(signal, 2);
    end

    % Normalize audio amplitude
    signal = signal / max(abs(signal));

    % Apply pre-emphasis filter to boost high frequencies
    signal = filter([1 -0.95], 1, signal);

    % Extract MFCC (Mel Frequency Cepstral Coefficients)
    coeffs = mfcc(signal, fs, 'NumCoeffs', 13);

    % Take mean of MFCC coefficients across frames
    feat = mean(coeffs, 1);
end