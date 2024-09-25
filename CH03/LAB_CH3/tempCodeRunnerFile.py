plt.plot(x, central_diff(g_func, x, h), color='r', label="Noiseless")
# # Plot the numerical derivatives using different h
# for hv in [h/2, h, 2*h]:
#     plt.plot(x, forward_diff(f_func, x, h), label=f"Forward diff, h={hv}")
