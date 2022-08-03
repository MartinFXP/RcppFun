// source: https://stackoverflow.com/questions/35923787/fast-large-matrix-multiplication-in-r

// [[depends(RcppEigen)]]

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

//' Vector times matrix rows
//'
//' Multiply a vector to each row of a matrix
//' Alternative to t(t(m)*v) in pure R
//' @param m numeric rxc matrix
//' @param v numeric vector with length c
//' @return rxc matrix with values m[i,j]*v[j]
//' @export
// [[Rcpp::export(rng = false)]]
SEXP perRow(NumericMatrix m, NumericVector v)
{
    int r = m.nrow();
    int c = m.ncol();
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            m(i, j) = m(i, j)*v[j];
        }
    }
    return wrap(m);
}

//' @export
//' @import Rcpp RcppEigen
//' @useDynLib RcppFun, .registration=TRUE
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export(rng = false)]]
String concatenate(std::string x, std::string y)
{
    return wrap(x + ":" + y);
}

//' Extend data by degree 2
//'
//' All interaction terms (including quadratic) will be computed.
//'
//' @param x numeric nxm matrix with variables as columns
//' @param exp exponent (see polynomial kernel)
//' @param base base (see polynomial kernel)
//' @param coef0 factor (see polynomial kernel)
//' @return extended matrix with ((n-1)*n)/2 + 2n + 1 columns
//' @export
// [[Rcpp::export(rng = false)]]
SEXP extendData(NumericMatrix x, double exp = 0.5, double base = 2, double coef0 = 1)
{
    int nr = x.nrow(); // INTEGER(Rf_getAttrib(x, R_DimSymbol))[0];
    int nc = x.ncol(); // INTEGER(Rf_getAttrib(x, R_DimSymbol))[1];
    int nc2 = (nc * (nc - 1)) / 2 + 2 * nc + 1;
    NumericMatrix m(nr, nc2);
    CharacterVector cx = colnames(x);
    CharacterVector cm(nc2);
    double fac2 = pow(base * coef0, exp);
    double fac3 = pow(base, exp);
    int i, j;
    int count = 0;
    for (i = 0; i < nc; i++)
    {
        m(_, count) = x(_, i) * fac2;
        cm[count] = cx[i];
        count++;
        m(_, count) = x(_, i) * x(_, i);
        std::string s1{cx[i]};
        cm[count] = concatenate(s1, s1);
        count++;
        if (i == nc - 1)
        {
            break;
        }
        for (j = i + 1; j < nc; j++)
        {
            m(_, count) = x(_, i) * x(_, j) * fac3;
            std::string s2{cx[j]};
            cm[count] = concatenate(s1, s2);
            count++;
        }
    }
    m(_, count) = NumericVector(nr, coef0);
    cm[count] = "coef0";
    colnames(m) = cm;
    rownames(m) = rownames(x);
    return wrap(m);
}

//' Compare columns and vector
//'
//' Matches columns of a matrix against a vector.
//' @param x nxm character matrix
//' @param y character vector of length n
//' @param s string denoting the gaps in matrix and vector that
//' will not be used for comparison
//' @return Numeric matrix with a relative (regarding to the gap string)
//' best match for the column
//' in column one and a relative best match for the vector in column 2.
//' @export
// [[Rcpp::export(rng = false)]]
SEXP compareMV(CharacterMatrix x, CharacterVector y, std::string s = "-")
{
    int r = x.nrow();
    int c = x.ncol();
    double n, m, ln, lm;
    NumericMatrix a(c, 2);
    for (int i = 0; i < c; i++)
    {
        n = 0;
        m = 0;
        ln = 0;
        lm = 0;
        for (int j = 0; j < r; j++)
        {
            if (x(j, i) != s)
            {
                ln++;
                if (x(j, i) == y(j))
                {
                    n++;
                }
            }
            if (y(j) != s)
            {
                lm++;
                if (x(j, i) == y(j))
                {
                    m++;
                }
            }
        }
        a(i, 0) = n / ln;
        a(i, 1) = m / lm;
    }
    return wrap(a);
}

//' Matrix multiplication
//'
//' @param A numeric nxm matrix
//' @param B numeric mxl matrix
//' @return numeric nxl matrix
//' @export
// [[Rcpp::export(rng = false)]]
SEXP eigenMapMatMult(const Eigen::Map<Eigen::MatrixXd> A, Eigen::Map<Eigen::MatrixXd> B)
{
    Eigen::MatrixXd C = A * B;
    return wrap(C);
}

//' Transitive closure
//'
//' @param x numeric nxn matrix (should be binary)
//' @return transitive closure of x
//' @export
// [[Rcpp::export(rng = false)]]
SEXP transClose_W(NumericMatrix x)
{
    int nr = INTEGER(Rf_getAttrib(x, R_DimSymbol))[0];
    int nc = INTEGER(Rf_getAttrib(x, R_DimSymbol))[1];
    double *px = REAL(x);
    int i, j, k;
    for (k = 0; k < nr; k++)
    {
        for (i = 0; i < nc; i++)
        {
            for (j = 0; j < nr; j++)
            {
                px[i * nc + j] = (round(px[i * nc + j]) ||
                                  (round(px[i * nc + k]) && round(px[k * nc + j])));
            }
        }
    }
    return wrap(x);
}

//' Transitive closure after deletion
//'
//' @param x numeric nxn matrix (should be binary) that was transitively closed before an edge deletion
//' @param u row index of deleted edge
//' @param v column index of deleted edge
//' @return transitive closure of x
//' @export
// [[Rcpp::export(rng = false)]]
SEXP transClose_Del(NumericMatrix x, IntegerVector u, IntegerVector v)
{
    int nr = INTEGER(Rf_getAttrib(x, R_DimSymbol))[0];
    int nc = INTEGER(Rf_getAttrib(x, R_DimSymbol))[1];
    int i = v[0] - 1;
    int j = u[0] - 1;
    double *px = REAL(x);
    int k;
    for (k = 0; k < nr; k++)
    {
        px[i * nc + j] = (round(px[i * nc + j]) ||
                          (round(px[i * nc + k]) && round(px[k * nc + j])));
        if (px[i * nc + j] == 1)
        {
            break;
        }
    }
    return wrap(x);
}

//' Transitive closure after addition
//'
//' @param x numeric nxn matrix (should be binary) that was transitively closed before an edge addition
//' @param u row index of added edge
//' @param v column index of added edge
//' @return transitive closure of x
//' @export
// [[Rcpp::export(rng = false)]]
SEXP transClose_Ins(NumericMatrix x, IntegerVector u, IntegerVector v)
{
    int nr = INTEGER(Rf_getAttrib(x, R_DimSymbol))[0];
    int nc = INTEGER(Rf_getAttrib(x, R_DimSymbol))[1];
    int k = u[0] - 1;
    int l = v[0] - 1;
    double *px = REAL(x);
    int i, j;
    for (i = 0; i < nc; i++)
    {
        for (j = 0; j < nr; j++)
        {
            px[j * nc + i] = (round(px[j * nc + i]) ||
                              (round(px[j * nc + l]) && round(px[k * nc + i])));
        }
    }
    return wrap(x);
}

//' Maximum per row
//'
//' @param x numeric nxm matrix
//' @return numeric vector of length n with the maximum for each row
//' @export
// [[Rcpp::export(rng = false)]]
SEXP maxCol_row(NumericMatrix x)
{
    int nr = INTEGER(Rf_getAttrib(x, R_DimSymbol))[0];
    int nc = INTEGER(Rf_getAttrib(x, R_DimSymbol))[1];
    double *px = REAL(x), *buf = (double *)R_alloc(nr, sizeof(double));
    for (int i = 0; i < nr; i++)
        buf[i] = R_NegInf;
    SEXP ans = PROTECT(Rf_allocVector(INTSXP, nr));
    int *pans = INTEGER(ans);
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            if (px[i + j * nr] > buf[i])
            {
                buf[i] = px[i + j * nr];
                pans[i] = j + 1;
            }
        }
    }
    UNPROTECT(1);
    return wrap(ans);
}
