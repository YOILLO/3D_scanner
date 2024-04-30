import math
import os, torch, time, shutil, json, glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict

import open3d
from matplotlib.pyplot import hist

import point_cloud_2
from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed, load_config
# from lib.tester import get_trainer
from lib.loss import MetricLoss
from sklearn.metrics import accuracy_score
from configs.models import architectures

from torch import optim
from torch import nn

setup_seed(0)

# from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
from point_cloud import PointCloudsNN
from point_cloud_2 import PointCloudsСNN

print(torch.cuda.is_available())
print(torch.cuda.device(0))


def normalize(tgt, src, trans):
    mx = max(torch.max(torch.abs(tgt)), torch.max(torch.abs(src)), torch.max(torch.abs(trans)))
    return torch.div(tgt, mx), torch.div(src, mx), torch.div(trans, mx)


def result_to_arr(res):
    rot = [[res[0].item()],
           [res[1].item()],
           [res[2].item()]]
    trans = [[res[3].item()],
             [res[4].item()],
             [res[5].item()]]
    return rot, trans


def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def draw_point_clouds(src_pcd, tgt_pcd, rot, trans, name):
    src_pcd_o3d = open3d.geometry.PointCloud()
    src_pcd_o3d.points = open3d.utility.Vector3dVector(src_pcd)
    src_pcd_o3d.paint_uniform_color([1, 0, 0])
    mesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
    rotat = rot * 2 * math.pi
    rat_matr = mesh.get_rotation_matrix_from_xyz(rotat)
    src_pcd_o3d.rotate(rat_matr, np.array([[0], [0], [0]]))
    src_pcd_o3d.translate(trans)

    tgt_pcd_o3d = open3d.geometry.PointCloud()
    tgt_pcd_o3d.points = open3d.utility.Vector3dVector(tgt_pcd)
    tgt_pcd_o3d.paint_uniform_color([0, 1, 0])

    # open3d.visualization.draw_geometries([src_pcd_o3d, tgt_pcd_o3d])

    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(src_pcd_o3d)
    vis.update_geometry(src_pcd_o3d)
    vis.add_geometry(tgt_pcd_o3d)
    vis.update_geometry(tgt_pcd_o3d)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"history2/{name}.png", do_render=True)
    vis.destroy_window()


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # backup the files
    os.system(f'cp -r models {config.snapshot_dir}')
    os.system(f'cp -r datasets {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    shutil.copy2('main.py', config.snapshot_dir)

    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    test_loader = torch.utils.data.DataLoader(val_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

    # train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
    #                                                           batch_size=1,
    #                                                           shuffle=True,
    #                                                           num_workers=config.num_workers,
    #                                                           )
    # val_loader, _ = get_dataloader(dataset=val_set,
    #                                       batch_size=1,
    #                                       shuffle=False,
    #                                       num_workers=1,
    #                                       neighborhood_limits=neighborhood_limits
    #                                       )
    #
    # test_loader, _ = get_dataloader(dataset=benchmark_set,
    #                                        batch_size=1,
    #                                        shuffle=False,
    #                                        num_workers=1,
    #                                        neighborhood_limits=neighborhood_limits)

    width = 120
    length = 500
    backup_num = 0
    mod = 1

    if mod:
        model = PointCloudsСNN(width, length)
    else:
        model = PointCloudsNN(width)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)  # lr = 0.0015

    trainLoss = []
    testLoss = []

    step = backup_num

    if backup_num > 0:
        model.load_state_dict(torch.load(f"history2/{mod}_point_cloud_model_{backup_num}_wd_{width}_{length}.mod"))

        with open(f"history2/{mod}_train_loss_{backup_num}_wd_{width}_{length}.txt", "r") as lossF:
            for line in lossF.readlines():
                trainLoss.append(float(line.strip()))

        with open(f"history2/{mod}_test_loss_{backup_num}_wd_{width}_{length}.txt", "r") as lossF:
            for line in lossF.readlines():
                testLoss.append(float(line.strip()))

    # tb = SummaryWriter(comment='Run simple cnn on mnist')
    # print(model)

    # for data in train_loader:
    # print(1, data["points"][0], len(data["points"]), data["points"][0].size())
    # print(2, data["neighbors"][0])
    # print(3, data["pools"][0], data["pools"][0].size())
    # print(4, data["upsamples"][0])
    # print(5, data["features"][0])
    # print(6, data["stack_lengths"][0])
    # print(7, data["rot"][0])
    # print(8, data["trans"][0])
    # print(9, data["correspondences"][0])
    # print(10, data["src_pcd_raw"], data["src_pcd_raw"].size())
    # print(11, data["tgt_pcd_raw"], data["tgt_pcd_raw"].size())
    # print(12, data["sample"])
    # input()
    step_rel = 0
    epoch = 0
    while step_rel + len(train_loader) < backup_num:
        epoch += 1
        step_rel += len(train_loader)

    while epoch <= 10:
        epoch += 1
        running_loss = 0
        running_loss_num = 0
        for src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, _ in train_loader:
            step_rel += 1
            if step_rel < backup_num:
                print(step_rel, backup_num, len(train_loader))
                continue
            step += 1
            # if step % 2 != 0:
            # continue

            tgt_pcd, src_pcd, trans = normalize(tgt_pcd, src_pcd, trans)

            if mod:
                while (len(tgt_pcd[0]) % length):
                    tgt_pcd = torch.cat((tgt_pcd, torch.from_numpy(np.array([[[0.0, 0.0, 0.0]]], np.float32))), dim=1)

                while (len(src_pcd[0]) % length):
                    src_pcd = torch.cat((src_pcd, torch.from_numpy(np.array([[[0.0, 0.0, 0.0]]], np.float32))), dim=1)

            optimizer.zero_grad()

            result = model.forward(src_pcd.float(), tgt_pcd.float())

            rotation = rotationMatrixToEulerAngles(rot[0])
            rotation = rotation / (2 * math.pi)
            rotation = np.array(rotation, np.float32)

            data_res = torch.from_numpy(np.array([rotation[0], rotation[1], rotation[2],
                                                  trans[0][0][0], trans[0][1][0], trans[0][2][0]], np.float32))

            loss = loss_func(result, data_res)
            loss.backward()
            optimizer.step()

            print(epoch, loss.item(), step)
            running_loss += loss.item()
            running_loss_num += 1

            if step % 1000 == 0:
                test_loss = 0
                test_loss_num = 0

                # Set mode withou gradient calculation
                with torch.no_grad():
                    model.eval()

                    # Run validation on test dataset
                    step2 = 0
                    save_screen = False
                    print("Start testing")
                    for src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, _ in test_loader:
                        print(len(test_loader), step2)
                        step2 += 1
                        #if step2 % 6 != 0:
                            #continue

                        tgt_pcd, src_pcd, trans = normalize(tgt_pcd, src_pcd, trans)

                        if mod:
                            while len(tgt_pcd[0]) % length:
                                tgt_pcd = torch.cat(
                                    (tgt_pcd, torch.from_numpy(np.array([[[0.0, 0.0, 0.0]]], np.float32))), dim=1)

                            while len(src_pcd[0]) % length:
                                src_pcd = torch.cat(
                                    (src_pcd, torch.from_numpy(np.array([[[0.0, 0.0, 0.0]]], np.float32))), dim=1)

                        result = model.forward(src_pcd.float(), tgt_pcd.float())

                        rotation = rotationMatrixToEulerAngles(rot[0]) / (2 * math.pi)

                        data_res = torch.from_numpy(np.array([rotation[0], rotation[1], rotation[2],
                                                              trans[0][0][0], trans[0][1][0], trans[0][2][0]]))

                        if not save_screen:
                            rot_t, trans_t = result_to_arr(result)
                            #draw_point_clouds(src_pcd.numpy()[0], tgt_pcd.numpy()[0], np.array(rotation, np.float32),
                                              #np.array(trans[0], np.float32),
                                              #f"{mod}_screen_aim_{step}_{width}_{length}_{step2}")
                            draw_point_clouds(src_pcd.numpy()[0], tgt_pcd.numpy()[0], np.array(rot_t, np.float32),
                                              np.array(trans_t, np.float32), f"{mod}_screen_{step}_{width}_{length}_{step2}")
                            save_screen = True

                        testsample_loss = loss_func(result, data_res)
                        test_loss += testsample_loss.item()
                        test_loss_num += 1

                        ps = torch.exp(result)

                        # top_p, top_class = ps.topk(1, dim=1)

                        # accuracy += accuracy_score(data_res.view(*top_class.shape).cpu(), top_class.cpu())

                # Returnn train mode
                model.train()

                trainLoss.append(running_loss / running_loss_num)
                running_loss_num = 0
                running_loss = 0
                testLoss.append(test_loss / test_loss_num)
                # accHistory.append(accuracy / len(test_loader))
                train_f = open(f"history2/{mod}_train_loss_{step}_wd_{width}_{length}.txt", "w")
                for i in trainLoss:
                    train_f.write(str(i))
                    train_f.write("\n")
                train_f.close()

                test_f = open(f"history2/{mod}_test_loss_{step}_wd_{width}_{length}.txt", "w")
                for i in testLoss:
                    test_f.write(str(i))
                    test_f.write("\n")
                test_f.close()

                # acc_f = open(f"history2/acc_{step}.txt", "w")
                # for i in accHistory:
                # acc_f.write(str(i))
                # acc_f.write("\n")
                # acc_f.close()

                torch.save(model.state_dict(), f"history2/{mod}_point_cloud_model_{step}_wd_{width}_{length}.mod")
                torch.onnx.export(model.final,
                                  (torch.randn(width, requires_grad=True), torch.randn(width, requires_grad=True)),
                                  f"history2/{mod}_point_cloud_model_final_{step}_wd_{width}_{length}.onnx",
                                  input_names=["src", "tgt"],
                                  output_names=["transform"],
                                  do_constant_folding=True)

                if mod:
                    torch.onnx.export(model.encoder1.layer,
                                      (torch.randn(width, requires_grad=True), torch.randn((1, length, 3), requires_grad=True)),
                                      f"history2/{mod}_point_cloud_model_enc1_{step}_wd_{width}_{length}.onnx",
                                      input_names=["tmp", "point"],
                                      output_names=["transform"],
                                      do_constant_folding=True)

                    torch.onnx.export(model.encoder2.layer,
                                      (torch.randn(width, requires_grad=True), torch.randn((1, length, 3), requires_grad=True)),
                                      f"history2/{mod}_point_cloud_model_enc2_{step}_wd_{width}_{length}.onnx",
                                      input_names=["tmp", "point"],
                                      output_names=["transform"],
                                      do_constant_folding=True)

                else:
                    torch.onnx.export(model.encoder1.layer,
                                      (torch.randn(width, requires_grad=True), torch.randn(3, requires_grad=True)),
                                      f"history2/{mod}_point_cloud_model_enc1_{step}_wd_{width}.onnx",
                                      input_names=["tmp", "point"],
                                      output_names=["transform"],
                                      do_constant_folding=True)

                    torch.onnx.export(model.encoder2.layer,
                                      (torch.randn(width, requires_grad=True), torch.randn(3, requires_grad=True)),
                                      f"history2/{mod}_point_cloud_model_enc2_{step}_wd_{width}.onnx",
                                      input_names=["tmp", "point"],
                                      output_names=["transform"],
                                      do_constant_folding=True)
