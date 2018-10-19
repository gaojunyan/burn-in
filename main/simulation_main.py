import simulation.simulation_tool as tool
from utils import file_tool
import simulation.simulation_visual as vis

if __name__ == "__main__":
    weak_train_n, normal_train_n = 1000, 1000
    weak_test_n, normal_test_n = 50, 50

    t_n, sdt = 80, 50
    # t_n, sdt = 50,3

    savefile, process = '../data/gamma', 'gamma'
    # savefile, process = '../data/wiener', 'wiener'
    file_tool.mkdir(savefile)

    tool.generate_simulation_data(weak_train_n=weak_train_n, normal_train_n=normal_train_n,
                                  weak_test_n=weak_test_n, normal_test_n=normal_test_n,
                                  t_n=t_n, dt=sdt, save_file=savefile, process=process)

    vis.visualize(weak_train_n, normal_train_n, t_n, sdt,
                  '{}/{}_data_train.pkl'.format(savefile, process),
                  '{}/{}_data_train.jpg'.format(savefile, process))

    vis.visualize(weak_test_n, normal_test_n, t_n, sdt,
                  '{}/{}_data_test.pkl'.format(savefile, process),
                  '{}/{}_data_test.jpg'.format(savefile, process))
