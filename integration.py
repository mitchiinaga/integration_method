import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

def acceleration(pos):
    r_inv = 1.0 / np.linalg.norm(pos)
    a = pos * (-r_inv**3)
    return a

def jerk(pos, vel):
    r_inv = 1.0 / np.linalg.norm(pos)
    j1 = vel * (-r_inv**3)
    j2 = pos * (3 * np.dot(pos, vel) * r_inv**5)
    return j1 + j2

class Particle:
    def __init__(self, pos = [0.0, 0.0], vel = [0.0, 0.0]):
        self.pos = np.array(pos)
        self.vel = np.array(vel)

class Integrator:
    def euler(self, pos, vel, dt):
        a = acceleration(pos)
        pos += vel * dt
        vel += a * dt

    def symplectic_euler(self, pos, vel, dt):
        a = acceleration(pos)
        vel += a * dt
        pos += vel * dt

    def rk2(self, pos, vel, dt):
        a0 = acceleration(pos)
        p1 = pos + vel * dt
        v1 = vel + a0 * dt
        a1 = acceleration(p1)
        pos += (vel + v1) * (dt * 0.5)
        vel += (a0 + a1) * (dt * 0.5)

    def leapfrog(self, pos, vel, dt):
        vel += acceleration(pos) * (0.5 * dt)
        pos += vel * dt
        vel += acceleration(pos) * (0.5 * dt)

    def rk4(self, pos, vel, dt):
        p = []
        v = []
        a = []
        
        p.append(pos)
        v.append(vel)
        a.append(acceleration(p[-1]))

        p.append(p[0] + v[-1] * (dt * 0.5))
        v.append(v[0] + a[-1] * (dt * 0.5))
        a.append(acceleration(p[-1]))

        p.append(p[0] + v[-1] * (dt * 0.5))
        v.append(v[0] + a[-1] * (dt * 0.5))
        a.append(acceleration(p[-1]))

        p.append(p[0] + v[-1] * dt)
        v.append(v[0] + a[-1] * dt)
        a.append(acceleration(p[-1]))

        pos += (v[0] + 2.0 * v[1] + 2.0 * v[2] + v[3]) * (dt / 6.0)
        vel += (a[0] + 2.0 * a[1] + 2.0 * a[2] + a[3]) * (dt / 6.0)

    def hermite(self, pos, vel, dt):
        a0 = acceleration(pos)
        j0 = jerk(pos, vel)
        pos += vel * dt + a0 * (dt**2 * 0.5) + j0 * (dt**3 / 6.0)
        vel += a0 * dt + j0 * (dt**2 * 0.5)

        a1 = acceleration(pos)
        j1 = jerk(pos, vel)
        a2 = -6.0 * (a0 - a1) - dt * (4.0 * j0 + 2.0 * j1)
        a3 = 12.0 * (a0 - a1) + 6.0 * dt * (j0 + j1)
        pos += a2 * (dt**2 / 24.0) + a3 * (dt**2 / 120.0)
        vel += a2 * (dt / 6.0) + a3 * (dt / 24.0)

    def __init__(self, method = "euler"):
        if method == "euler":
            self.method = Integrator.euler
        elif method == "symplectic_euler":
            self.method = Integrator.symplectic_euler
        elif method == "rk2":
            self.method = Integrator.rk2
        elif method == "rk4":
            self.method = Integrator.rk4
        elif method == "leapfrog":
            self.method = Integrator.leapfrog
        elif method == "hermite":
            self.method = Integrator.hermite
        else:
            print(method, "is invalid.")
            sys.exit()

    def run(self, particle, dt):
        self.method(self, particle.pos, particle.vel, dt)

def main():
    p = Particle([1.0, 0.0], [0.0, 0.2])

    method = sys.argv[1] if len(sys.argv) > 1 else "euler"

    itg = Integrator(method)
    time = 0.0
    dt = 1e-3

    # plot value
    plot0 = [p.pos[0]]
    plot1 = [p.pos[1]]

    while time < 150:
        time += dt
        itg.run(p, dt)

        plot0.append(p.pos[0])
        plot1.append(p.pos[1])

    plt.plot(plot0, plot1)
    plt.show()

if __name__ == "__main__":
    main()