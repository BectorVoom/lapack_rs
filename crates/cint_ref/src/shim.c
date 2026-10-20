/* Recording interposer for libcint 6.1.3's `_CINTdiagonalize`.
 *
 * `eigh.c` and `rys_wheeler.c` are compiled with
 * `-D_CINTdiagonalize=cintref_diagonalize_impl`, so the real solver keeps its
 * body under a new name and this file supplies `_CINTdiagonalize`. Driving
 * `CINTrys_jacobi` / `CINTrys_laguerre` therefore records the exact Wheeler
 * tridiagonals libcint hands its eigensolver — the A0 corpus — while still
 * returning the vendor's own answer to the caller.
 *
 * Recording is off unless `cintref_record_begin` has been called, and the
 * buffer is caller-owned, so the oracle stays a pure function by default. */

#include <string.h>

#define CINTREF_MAXN 13

int cintref_diagonalize_impl(int n, double *diag, double *diag_off1,
                             double *eig, double *vec);

/* Recording state. Single-threaded by construction: the harness drives one
 * sweep at a time and the tests that use it are `--test-threads=1`. */
static int    cintref_recording = 0;
static int    cintref_count     = 0;
static int    cintref_cap       = 0;
static int    *cintref_ns       = 0;
static double *cintref_diag     = 0;
static double *cintref_off      = 0;

void cintref_record_begin(int cap, int *ns, double *diag, double *off)
{
        cintref_recording = 1;
        cintref_count     = 0;
        cintref_cap       = cap;
        cintref_ns        = ns;
        cintref_diag      = diag;
        cintref_off       = off;
}

int cintref_record_end(void)
{
        cintref_recording = 0;
        return cintref_count;
}

int _CINTdiagonalize(int n, double *diag, double *diag_off1,
                     double *eig, double *vec)
{
        if (cintref_recording && cintref_count < cintref_cap) {
                int k = cintref_count;
                int i;
                cintref_ns[k] = n;
                for (i = 0; i < CINTREF_MAXN; i++) {
                        cintref_diag[k * CINTREF_MAXN + i] = 0.;
                        cintref_off[k * CINTREF_MAXN + i] = 0.;
                }
                for (i = 0; i < n && i < CINTREF_MAXN; i++) {
                        cintref_diag[k * CINTREF_MAXN + i] = diag[i];
                        cintref_off[k * CINTREF_MAXN + i] = diag_off1[i];
                }
                cintref_count = k + 1;
        }
        return cintref_diagonalize_impl(n, diag, diag_off1, eig, vec);
}

/* Direct access to the vendor solver, bypassing the recorder. */
int cintref_diagonalize(int n, double *diag, double *diag_off1,
                        double *eig, double *vec)
{
        return cintref_diagonalize_impl(n, diag, diag_off1, eig, vec);
}
