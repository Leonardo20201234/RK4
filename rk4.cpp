#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>

// ── Eigen: vectores y operaciones rápidas ────────────────────────────────────
#include <Eigen/Dense>

using Vec6 = Eigen::Matrix<double, 6, 1>;   // estado completo [x,y,z,vx,vy,vz]
using Vec3 = Eigen::Matrix<double, 3, 1>;

// ─────────────────────────────────────────────────────────────────────────────
// Fuerza: equivalente a F = sp.Matrix([-x, 0, 0])
//   F(t, X, V) = (-X[0], 0, 0)
// ─────────────────────────────────────────────────────────────────────────────
inline Vec3 fuerza(double /*t*/, const Vec3& X, const Vec3& /*V*/)
{
    return Vec3(-X(0), 0.0, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// sistema(t, Y) — equivalente a def sistema(t, Y)
//   dX/dt = V
//   dV/dt = F(t, X, V)
// ─────────────────────────────────────────────────────────────────────────────
inline Vec6 sistema(double t, const Vec6& Y)
{
    Vec3 X = Y.head<3>();
    Vec3 V = Y.tail<3>();

    Vec6 dY;
    dY.head<3>() = V;
    dY.tail<3>() = fuerza(t, X, V);
    return dY;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    using Clock = std::chrono::high_resolution_clock;

    // ── Parámetros (idénticos al Python) ────────────────────────────────────
    constexpr int    N        = 1'000'000;
    constexpr double tinicial = 0.0;
    constexpr double tfinal   = 6.0;
    constexpr double h        = (tfinal - tinicial) / N;

    // ── Reservar arreglos ────────────────────────────────────────────────────
    // Para N=1M, guardar toda la trayectoria ocupa ~48 MB (6 doubles × 1M)
    // Usamos Eigen::MatrixXd: filas = pasos, cols = 6
    Eigen::MatrixXd Sol(N, 6);   // Sol(i, :) = [x, y, z, vx, vy, vz]
    Eigen::VectorXd t_arr(N);

    // ── Condiciones iniciales ────────────────────────────────────────────────
    Sol.row(0) << 0.0, 0.0, 0.0,   // X(0)
                  1.0, 0.0, 0.0;   // V(0)
    t_arr(0) = tinicial;

    // ── RK4 ─────────────────────────────────────────────────────────────────
    std::cout << "Integrando " << N << " pasos (t: "
              << tinicial << " → " << tfinal << ", h=" << h << ")...\n";

    auto t_start = Clock::now();

    for (int i = 0; i < N - 1; ++i) {
        const double ti = t_arr(i);
        const Vec6   Yi = Sol.row(i).transpose();

        Vec6 k1 = sistema(ti,       Yi            );
        Vec6 k2 = sistema(ti + h/2, Yi + (h/2)*k1);
        Vec6 k3 = sistema(ti + h/2, Yi + (h/2)*k2);
        Vec6 k4 = sistema(ti + h,   Yi + h*k3     );

        Sol.row(i+1) = (Yi + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)).transpose();
        t_arr(i+1)   = ti + h;
    }

    auto t_end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Tiempo RK4: " << ms << " ms  ("
              << ms/N*1000.0 << " µs/paso)\n\n";

    // ── Verificación vs solución exacta x(t) = sin(t) ───────────────────────
    double error_max = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(Sol(i, 0) - std::sin(t_arr(i)));
        if (err > error_max) error_max = err;
    }
    std::cout << "Error máximo |x_RK4 - sin(t)|: " << error_max << "\n\n";

    // ── Guardar CSV (mismo formato que el Python exportaría) ─────────────────
    std::cout << "Guardando sol.csv...\n";
    auto tw0 = Clock::now();
    {
        std::ofstream csv("sol.csv");
        csv << "t,x,y,z,vx,vy,vz\n";
        csv << std::scientific << std::setprecision(10);

        // Escribir cada N/1000-ésimo punto para un CSV manejable (1001 filas)
        // Cambiar a `i++` si se quieren todos los N puntos
        int stride = N / 1000;
        for (int i = 0; i < N; i += stride) {
            csv << t_arr(i)   << ","
                << Sol(i, 0)  << ","
                << Sol(i, 1)  << ","
                << Sol(i, 2)  << ","
                << Sol(i, 3)  << ","
                << Sol(i, 4)  << ","
                << Sol(i, 5)  << "\n";
        }
    }
    auto tw1 = Clock::now();
    double ms_csv = std::chrono::duration<double, std::milli>(tw1 - tw0).count();
    std::cout << "CSV escrito en " << ms_csv << " ms\n";
    std::cout << "\nPara graficar (requiere gnuplot):\n"
              << "  gnuplot -e \"\n"
              << "    set datafile separator ',';\n"
              << "    set key outside; set grid; set xlabel 't';\n"
              << "    plot 'sol.csv' u 1:2 w l lc rgb 'red'   t 'x(t)',\\\n"
              << "         ''        u 1:3 w l lc rgb 'blue'  t 'y(t)',\\\n"
              << "         ''        u 1:4 w l lc rgb 'green' t 'z(t)';\n"
              << "    pause -1\"\n";

    return 0;
}