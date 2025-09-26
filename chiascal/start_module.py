import os
import sys
from collections import namedtuple


def init_proj_folder(model_version='v01'):
    proj_path = os.getcwd()
    proj = os.path.split(proj_path)[1]
    PathSet = namedtuple(
        'PathSet',
        '''proj_path data_path share_data_path ver_data_path deploy_data_path
        monitor_data_path raw_data_path processed_data_path
        intermediate_data_path report_set_path report_path model_set_path
        model_path final_model_path dss_path notebook_set_path notebook_path
        ''')
    data_path = os.path.join(proj_path, 'data')
    share_data_path = os.path.join(data_path, 'share')
    ver_data_path = os.path.join(data_path, model_version)
    deploy_data_path = os.path.join(data_path, 'deploy')
    monitor_data_path = os.path.join(data_path, 'monitor')
    raw_data_path = os.path.join(ver_data_path, 'raw')
    processed_data_path = os.path.join(ver_data_path, 'processed')
    intermediate_data_path = os.path.join(ver_data_path, 'intermediate')

    report_set_path = os.path.join(proj_path, 'reports')
    report_path = os.path.join(report_set_path, model_version)

    model_set_path = os.path.join(proj_path, 'models')
    model_path = os.path.join(model_set_path, model_version)
    final_model_path = os.path.join(model_set_path, 'final')

    dss_path = os.path.join(r'/app-data/xxxx', proj)

    notebook_set_path = os.path.join(proj_path, 'notebooks')
    notebook_path = os.path.join(notebook_set_path, model_version)

    SrcPath = namedtuple(
        'SrcPath',
        '''src_path src_data_path src_feature_path src_model_path
        src_report_path src_viz_path
        ''')
    src_path = os.path.join(proj_path, 'src')
    src_data_path = os.path.join(src_path, 'data')
    src_feature_path = os.path.join(src_path, 'feature')
    src_model_path = os.path.join(src_path, 'model')
    src_report_path = os.path.join(src_path, 'report')
    src_viz_path = os.path.join(src_path, 'viz')

    path_set = PathSet(
        proj_path, data_path, share_data_path, ver_data_path, deploy_data_path,
        monitor_data_path ,raw_data_path ,processed_data_path,
        intermediate_data_path, report_set_path, report_path, model_set_path,
        model_path, final_model_path, dss_path, notebook_set_path,
        notebook_path, src_path
    )
    src_path_set = SrcPath(
        src_path, src_data_path, src_feature_path, src_model_path,
        src_report_path, src_viz_path
    )
    for k, fl in path_set._asdict().items():
        if not os.path.exists(fl) and k != 'dss_path':
            os.mkdir(fl)
            if fl == src_path:
                with open(os.path.join(src_path, '__init__.py'), 'w') as f:
                    pass
    for k, fl in src_path_set._asdict().items():
        if not os.path.exists(fl):
            os.mkdir(fl)
            with open(os.path.join(fl, '__init__.py'), 'w') as f:
                f.write("import pkgutil\n")
                f.write("import importlib\n")
                f.write("pkg_name = __name__\n")
                f.write("pkg_1 = importlib.import_module(pkg_name)\n")
                f.write("submod_names = [name for _, name, _ in pkgutil.iter_modules(pkg_1.__path__)]\n")
                f.write("for submod_name in submod_names:\n")
                f.write("    submod = importlib.import_module(f'{pkg_name}.{submod_name}')\n")
                f.write("    funcs = [f for f in dir(submod) if f not in [\n")
                f.write("        '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',\n")
                f.write("        '__name__', '__package__', '__spec__']]\n")
                f.write("    for f in funcs:\n")
                f.write("        locals()[f] = getattr(submod, f)\n")
                f.write("    del funcs\n")
                f.write("    del submod\n")
                f.write("    del submod_name\n")
                f.write("    del f\n")
                f.write("del submod_names\n")
                f.write("del pkg_1\n")
                f.write("del pkg_name\n")
                f.write("del pkgutil\n")
                f.write("del importlib\n")
    return path_set


def get_proj_folder():
    note_path, model_version = os.path.split(os.getcwd())
    proj_path = os.path.split(note_path)[0]
    proj = os.path.split(proj_path)[1]
    PathSet = namedtuple(
        'PathSet',
        '''proj_path data_path share_data_path ver_data_path deploy_data_path
        monitor_data_path raw_data_path processed_data_path
        intermediate_data_path report_set_path report_path model_set_path
        model_path final_model_path dss_path notebook_set_path notebook_path
        ''')
    data_path = os.path.join(proj_path, 'data')
    share_data_path = os.path.join(data_path, 'share')
    ver_data_path = os.path.join(data_path, model_version)
    deploy_data_path = os.path.join(data_path, 'deploy')
    monitor_data_path = os.path.join(data_path, 'monitor')
    raw_data_path = os.path.join(ver_data_path, 'raw')
    processed_data_path = os.path.join(ver_data_path, 'processed')
    intermediate_data_path = os.path.join(ver_data_path, 'intermediate')

    report_set_path = os.path.join(proj_path, 'reports')
    report_path = os.path.join(report_set_path, model_version)

    model_set_path = os.path.join(proj_path, 'models')
    model_path = os.path.join(model_set_path, model_version)
    final_model_path = os.path.join(model_set_path, 'final')

    dss_path = os.path.join(r'/app-data/xxxx', proj)

    notebook_set_path = os.path.join(proj_path, 'notebooks')
    notebook_path = os.path.join(notebook_set_path, model_version)

    SrcPath = namedtuple(
        'SrcPath',
        '''src_path src_data_path src_feature_path src_model_path
        src_report_path src_viz_path
        ''')
    src_path = os.path.join(proj_path, 'src')
    src_data_path = os.path.join(src_path, 'data')
    src_feature_path = os.path.join(src_path, 'feature')
    src_model_path = os.path.join(src_path, 'model')
    src_report_path = os.path.join(src_path, 'report')
    src_viz_path = os.path.join(src_path, 'viz')

    path_set = PathSet(
        proj_path, data_path, share_data_path, ver_data_path, deploy_data_path,
        monitor_data_path ,raw_data_path ,processed_data_path,
        intermediate_data_path, report_set_path, report_path, model_set_path,
        model_path, final_model_path, dss_path, notebook_set_path,
        notebook_path, src_path
    )
    return path_set


if __name__ == '__main__':
    params = sys.argv[1:]
    proj_ = params[0]
    ver = 'v01'
    if len(params) > 1:
        ver = params[1]
    init_proj_folder(proj_, ver)
