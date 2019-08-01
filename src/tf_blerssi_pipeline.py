#!/usr/bin/env python3

import kfp.dsl as dsl
from kubernetes import client as k8s_client


def blerssi_train_op(tf_model_dir: str,
                   tf_export_dir: str, train_steps: int, batch_size: int,
                   learning_rate: float, step_name='blerssi-training'):
    return dsl.ContainerOp(
        name=step_name,
        image='docker.io/premkarthi/blerssi:v7',
        arguments=[
            '/opt/blerssi-model.py', 
            '--tf-model-dir', tf_model_dir,
            '--tf-export-dir', tf_export_dir,
            '--tf-train-steps', train_steps,
            '--tf-batch-size', batch_size,
            '--tf-learning-rate', learning_rate,
        ],
        file_outputs={'export': '/tf_export_dir.txt'}
    )


def kubeflow_deploy_op(tf_export_dir:str, server_name: str, pvc_name: str, step_name='serve'):
    return dsl.ContainerOp(
        name=step_name,
        image='docker.io/premkarthi/blerssi-deploy:v10',
        arguments=[
            '--cluster-name', 'blerssi-pipeline',
            '--model-export-path', tf_export_dir,
            '--server-name', server_name,
            '--pvc-name', pvc_name,
        ]
    )

def kubeflow_web_ui_op(step_name='web-ui'):
    return dsl.ContainerOp(
        name='web-ui',
        image='docker.io/premkarthi/tf-deploy-service:v7',
        arguments=[
            '--image', 'docker.io/premkarthi/blerssi-webapp:v3',
            '--name', 'web-ui',
            '--container-port', '9000',
            '--service-port', '80',
            '--service-type', "LoadBalancer"
        ]
    )

def kubeflow_tensorboard_op(tensorboard_dir, server_name: str, pvc_name: str, step_name='tensorboard'):
    return dsl.ContainerOp(
        name = step_name,
        image='docker.io/premkarthi/tf-tensorboard-deploy-service:v10',
        arguments=[
            '--image', 'docker.io/premkarthi/tensorboard:v10',
            '--name', 'tensorboard',
            '--logdir', tensorboard_dir,
            '--container-port', '6006',
            '--service-port', '80',
            '--service-type', "LoadBalancer"
        ]
    )

@dsl.pipeline(
    name='TF BLERSSI Pipeline',
    description='BLERSSI Pipelines for on-prem cluster'
)
def tf_blerssi_pipeline(
        model_name='blerssi',
        pvc_name='nfs',
        tf_model_dir='model',
        tf_export_dir='model/export',
        training_steps=200,
        batch_size=100,
        learning_rate=0.01):


    nfs_pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name='nfs')
    nfs_volume = k8s_client.V1Volume(name='nfs', persistent_volume_claim=nfs_pvc)
    nfs_volume_mount = k8s_client.V1VolumeMount(mount_path='/mnt/', name='nfs')
    tensorboard_dir='model'
    blerssi_training = blerssi_train_op(
        '/mnt/%s' % tf_model_dir,
        '/mnt/%s' % tf_export_dir,
        training_steps,
        batch_size,
        learning_rate)
    blerssi_training.add_volume(nfs_volume)
    blerssi_training.add_volume_mount(nfs_volume_mount)

    serve = kubeflow_deploy_op('/mnt/%s' % tf_export_dir, model_name, pvc_name)
    serve.add_volume(nfs_volume)
    serve.add_volume_mount(nfs_volume_mount)
    serve.after(blerssi_training)

    web_ui = kubeflow_web_ui_op()
    web_ui.add_volume(nfs_volume)
    web_ui.add_volume_mount(nfs_volume_mount)
    web_ui.after(serve)

    tensorboard = kubeflow_tensorboard_op('/mnt/%s' % tensorboard_dir,  model_name, pvc_name)
    tensorboard.add_volume(nfs_volume)
    tensorboard.add_volume_mount(nfs_volume_mount)
    tensorboard.after(blerssi_training)

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(tf_blerssi_pipeline, __file__+'.tar.gz')
