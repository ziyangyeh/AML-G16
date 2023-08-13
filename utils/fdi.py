import numpy as np
import vedo
import vtk


def get_point_ids_by_point_label(mesh, label_num=0, FDI=False):
    labels = mesh.pointdata["InsLabels"] if not FDI else mesh.pointdata["FDI"]
    index = np.where(labels == label_num)
    return index[0]


def get_cell_ids_by_point_index(mesh, indices):
    cell_ids = vtk.vtkIdList()
    data = mesh.inputdata()
    data.BuildLinks()
    result = []
    for i in np.unique(indices):
        data.GetPointCells(i, cell_ids)
        for j in range(cell_ids.GetNumberOfIds()):
            result.append(cell_ids.GetId(j))
    return result


def get_cell_ids_by_point_label(mesh, label_num=0, FDI=False):
    cell_ids = vtk.vtkIdList()
    data = mesh.inputdata()
    data.BuildLinks()
    indices = get_point_ids_by_point_label(mesh, label_num, FDI)
    result = []
    for i in np.unique(indices):
        data.GetPointCells(i, cell_ids)
        for j in range(cell_ids.GetNumberOfIds()):
            result.append(cell_ids.GetId(j))
    return result


def generate_cell_labels_from_point_label(mesh, FDI=False):
    cell_labels = np.zeros(mesh.ncells, dtype=np.uint8)
    if not FDI:
        for i in range(mesh.pointdata["InsLabels"].max() + 1):
            cell_labels[get_cell_ids_by_point_label(mesh, i, FDI)] = i
    else:
        for i in np.unique(mesh.pointdata["FDI"]):
            cell_labels[get_cell_ids_by_point_label(mesh, i, FDI)] = i
    return cell_labels


def crop_cells_by_point_label(mesh: vedo.Mesh, label_num=0, FDI=False):
    labels = mesh.pointdata["InsLabels"] if not FDI else mesh.pointdata["FDI"]
    ids = np.where(labels != label_num)
    return mesh.clone(deep=True).delete_cells_by_point_index(ids)


def get_fdi_label_map():
    label_odd = np.arange(1, 17, 2)
    label_even = np.arange(2, 17, 2)

    fdi1 = np.arange(1, 9) + 10
    fdi2 = np.arange(1, 9) + 20
    fdi3 = np.arange(1, 9) + 30
    fdi4 = np.arange(1, 9) + 40
    fdi_map = {}
    for k, v in zip(fdi1, label_even):
        fdi_map[k] = v
    for k, v in zip(fdi3, label_even):
        fdi_map[k] = v
    for k, v in zip(fdi2, label_odd):
        fdi_map[k] = v
    for k, v in zip(fdi4, label_odd):
        fdi_map[k] = v
    fdi_map[0] = 0

    return fdi_map
