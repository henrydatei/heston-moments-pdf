#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <map>

using namespace Eigen;
using namespace std;

const int RM_NP_return = 5;

VectorXd arrayToVectorXd(const double* arr, int size) {
    VectorXd vec(size);
    for (int i = 0; i < size; ++i) {
        vec(i) = arr[i];
    }
    return vec;
}

// Function to compute realized moments (simplified version)
MatrixXd rMomNP_return(const VectorXd& hf, int nM) {
    int N = hf.size();
    int tau = N / nM;
    VectorXd rt = hf.tail(N - tau);

    double m1r = rt.sum() / nM;
    double m2r_NP = (2 * (rt.array().exp() - 1 - rt.array())).sum() / (nM - 1);
    double m3r_NP = (6 * ((rt.array().exp() + 1) * rt.array() - 2 * (rt.array().exp() - 1))).sum() / (nM - 1);
    double m4r_NP = (12 * (rt.array().square() + 2 * (rt.array().exp() + 2) * rt.array() - 6 * (rt.array().exp() - 1))).sum() / (nM - 1);

    MatrixXd result(1, 4);
    result << m1r, m2r_NP, m3r_NP, m4r_NP;
    return result;
}

// Function to calculate cumulants
MatrixXd rCumulants(const VectorXd& hfData, int method, int months_overlap) {
    if (method == RM_NP_return) {
        return rMomNP_return(hfData, months_overlap);
    } else {
        cerr << "This method for computing cumulants is not implemented" << endl;
        return MatrixXd();
    }
}

extern "C"
{
void DoCalculate(double pDataIn[], int iInSize, double pDataOut[])
{
    VectorXd hfData = arrayToVectorXd(pDataIn, iInSize);
    
    int method = RM_NP_return;
    int months_overlap = 5;
    MatrixXd result = rCumulants(hfData, method, months_overlap);
    
    for (int i = 0; i < 4; ++i) {
        pDataOut[i] = result(0, i);
    }
}
}

// int main() {
//     VectorXd hfData(100);
//     hfData.setRandom();
    
//     int method = RM_NP_return;
//     int months_overlap = 5;
//     MatrixXd result = rCumulants(hfData, method, months_overlap);
    
//     cout << "Computed Cumulants:\n" << result << endl;
    
//     return 0;
// }
