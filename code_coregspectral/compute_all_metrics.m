function [Acc, nmi, F1, precision, recall, avgent, AR] = compute_all_metrics (True_label, Predict_label)

Acc = compute_acc(True_label,Predict_label);
[F1,precision,recall] = compute_f(True_label,Predict_label);

[A nmi avgent] = compute_nmi(True_label,Predict_label);

if (min(True_label)==0)
    [AR,RI,MI,HI]=RandIndex(True_label+1,Predict_label);
else
    [AR,RI,MI,HI]=RandIndex(True_label,Predict_label);
end