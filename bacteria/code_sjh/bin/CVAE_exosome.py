import os, numpy, csv, time
import torch,visdom
from torch import nn,optim
from torch.utils.data import DataLoader
from bacteria.code_sjh.utils import train_CVAE, Process, visdom_utils, projectroot,coderoot
from bacteria.code_sjh.utils import Validation as validation
from bacteria.code_sjh.models import BCE, CVAE2_Dlabel_Dclassifier, RamanData

def eval_CVAE(model, loader, device):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            _, logits, __ = model(x, y)
            pred = logits.argmax(dim = 1)
        c_t = torch.eq(pred, y.to(device)).sum().float().item()
        correct += c_t
    acc = correct / total
    return acc

def main(model, train_db, val_db, test_db,
         criteon,
         batchsz = 40, numworkers = 0, lr = 0.0001, viz = None,
         save_dir = os.path.join(coderoot, "checkpoints", ),
         epochs = 100,
         device = None,
         epo_interv = 30,
         rates = None,
         ):
    if rates is None:
        rates = dict(
            label_rate = 50.,
            kld_rate = 200.0,
        )
    if viz == None:
        viz = visdom.Visdom()

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
    viz.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k)))
    viz.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model.model_loaded = True
    # save_path = os.path.join(save_dir, model.model_name + ".mdl")
    # if not os.path.exists(save_path):
    # 	with open(save_path, "w", newline = ""):
    # 		pass

    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
                              )
    val_loader = DataLoader(val_db, batch_size = batchsz, num_workers = numworkers)
    test_loader = DataLoader(test_db, batch_size = batchsz, num_workers = numworkers)
    label2data_val = val_db.get_data_sorted_by_label()
    sample2data_val = val_db.get_data_sorted_by_sample()
    label2data_train = train_db.get_data_sorted_by_label()
    # label2data_train = train_db.get_data_sorted_by_sample()
    label2name = val_db.label2name()
    xs = torch.tensor(val_db.xs)
    for epoch in range(epochs):
        loss = train_CVAE(model, lr = lr, device = device, train_loader = train_loader, criteon = criteon,
                          optimizer = optimizer, idx = epoch, **rates)
        viz.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
        global_step += 1
        if epoch % epo_interv == epo_interv - 1 or epoch == epochs - 1:
            model.eval()

            val_acc = eval_CVAE(model, val_loader, device)
            # print("epoch:{},acc:{}".format(epoch, val_acc))
            if val_acc >= best_acc:
                model.save(os.path.join(save_dir, "trained", "epoch{}.mdl".format(epoch + 1)))
                model.Encoder.save(os.path.join(save_dir, "encoder", "epoch{}.mdl".format(epoch + 1)))
                best_acc = val_acc
                best_epoch = epoch+1
            viz.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
            train_acc = eval_CVAE(model, train_loader, device)
            viz.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
        if epoch % 1000 == 999 or epoch == epochs - 1:
            with torch.no_grad():
                newwin = 0
                for samplename in sample2data_val.keys():
                    spectrum = sample2data_val[samplename].to(device)
                    if model.neck_axis < 4:
                        model.neck_vis(spectrum, samplename,
                                       win = "VAE_val_neck_vis_samplewise_epoch:" + str(epoch + 1),
                                       vis = viz,
                                       update = None if newwin == 0 else "append", rand = 0.3)
                        newwin = 1
                for idx in range(len(label2name.keys())):

                    spectrum = label2data_val[idx]
                    spectrum = spectrum.to(device)
                    spectrum_hat = model(spectrum)[0]
                    win_main = "spectrum_VAE_val_" + label2name[idx] + "_epoch:" + str(epoch + 1)
                    win = win_main + label2name[idx]

                    validation.spectrum_vis(spectrum, xs, win, name = "x", vis = viz)
                    validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = viz)
                    if model.neck_axis < 4:
                        model.neck_vis(spectrum, label2name[idx], win = "VAE_val_neck_vis_epoch:" + str(epoch + 1),
                                       vis = viz,
                                       update = None if idx == 0 else "append")
                    # with torch.no_grad():
                    spectrum = label2data_train[idx]
                    spectrum = spectrum.to(device)
                    spectrum_hat = model(spectrum)[0]
                    win = "spectrum_VAE_train_" + label2name[idx] + "_epoch:" + str(epoch + 1)
                    validation.spectrum_vis(spectrum, xs, win, name = "x", vis = viz)
                    validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = viz)
                    if model.neck_axis < 4:
                        model.neck_vis(spectrum, label2name[idx], vis = viz,
                                       update = None if idx == 0 else "append",
                                       win = "VAE_train_neck_vis_epoch:" + str(epoch + 1))
    model.load(os.path.join(save_dir, "trained", "epoch{}.mdl".format(best_epoch)))
    test_acc = eval_CVAE(model, test_loader, device)

    return best_acc, test_acc


if __name__ == '__main__':
    visdom_utils.startVisdomServer()
    Model = CVAE2_Dlabel_Dclassifier
    tvt = [0.7, 0.2, 0.1]
    raman = RamanData.Raman_dirwise
    # dataroot = os.path.join(projectroot, "data", "data_AST")
    dataroot = os.path.join(projectroot, "data", "liver", "liver_all_samplewise")
    backend = ".csv"
    delimeter = ","
    dataformat = {"Wavelength": 0, "Column": 1, "Intensity": 2}
    readdatafunc0 = RamanData.getRamanFromFile(wavelengthstart = 596, wavelengthend = 1802, delimeter = delimeter,
                                               dataname2idx = dataformat)
    from scipy import interpolate


    def readdatafunc(filepath):
        R, X = readdatafunc0(filepath)
        f = interpolate.interp1d(X, R, kind = "cubic")
        newX = numpy.linspace(600, 1800, 760)
        newR = f(newX)
        return newR, newX


    config = dict(dataroot = dataroot, backEnd = backend, t_v_t = tvt, LoadCsvFile = readdatafunc)
    k_split = 5
    criteon = nn.BCELoss()

    bestaccs, testaccs = [], []
    f = open("VAErecord"+time.asctime().replace(":","-").replace(" ","_")+".csv","w",newline = "")
    writer = csv.writer(f)

    train_cfg = dict(
        epochs = 5000, batchsz = 256,
        lr = 0.0001, rates = dict(
            label_rate = 50.,
            kld_rate = 200.0,
        )
    )
    f.write(config.__str__()+"\n")
    f.write(train_cfg.__str__()+"\n")
    writer.writerow(["bestaccs", "testaccs"])
    for k in range(k_split):
        train_db = raman(**config, mode = "train", k = k)
        val_db = raman(**config, mode = "val", k = k)

        test_db = raman(**config, mode = "test", k = k)
        s_t, s_l = train_db[0]
        s_t = torch.unsqueeze(s_t, dim = 0)
        n_c = train_db.numclasses
        device = torch.device("cuda")
        model = Model(s_t, n_c).to(device)
        train_db.show_data(win = "train_db_data")
        val_db.show_data(win = "val_db_data")


        b, t = main(model, train_db, val_db, test_db, criteon, **train_cfg)
        bestaccs.append(b)
        testaccs.append(t)
        writer.writerow([b,t])
    ba = sum([ numpy.mean(numpy.array(bestaccs)), " +- ", numpy.std(numpy.array(bestaccs))])
    ta = sum([numpy.mean(numpy.array(testaccs)), " +- ", numpy.std(numpy.array(testaccs))])
    writer.writerow([ba,ta])
    f.close()

    print("best acc:",ba)
    print("test acc", ta)
