% new_params = converter(params);
% save("params.mat", "new_params")
% 
% new_state = converter(state);
% save("state.mat", "new_state")

new_DAEnet = converter_NN(DAEnet);
save("DAEnet.mat", "new_DAEnet")