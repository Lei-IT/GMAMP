% -----------------------------------------------------------------------
% If you use this code, please cite the paper below. Thank you.
%
% [1] F. Tian, L. Liu, and X. Chen, "Generalized Memory approximate message 
%     passing," arXiv preprint arXiv:2110.06069, Oct. 2021, [Online]
%     Available: https://arxiv.org/pdf/2110.06069.pdf
%
% [2] L. Liu, S. Huang, and B. M. Kurkoski, "Memory approximate message 
%     passing," arXiv preprint arXiv:2012.10861, Dec. 2020, [Online]
%     Available: https://arxiv.org/abs/2012.10861
%
%                                --by Feiyan Tian and Lei Liu, 2021    
% -----------------------------------------------------------------------
function [u_new, v_new] = Ortho(u_post, v_post, u_pri, v_pri)
    % Gaussian division
    v_reverse = v_post.^(-1) - v_pri.^(-1);
    v_reverse_u = v_post.^(-1) * u_post - v_pri.^(-1) * u_pri;  
    v_new = v_reverse.^(-1);
    u_new = v_new * v_reverse_u;
end

