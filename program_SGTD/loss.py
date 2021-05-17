def build_loss(logits_list, maps, masks, label):
    #logits_map = logits
    print('len-logits_list:', len(logits_list))
    logits_cla = get_logtis_cla_from_logits_list(logits_list)
    logits = logits_cla
    print('logits-shape:', logits.get_shape())

    label=tf.cast(label,tf.int32)
    print(label.get_shape())
    label=tf.one_hot(label,depth=num_classes,on_value=1.0,off_value=0.0,axis=-1)
    label=tf.squeeze(label, axis=1)
    print(label.get_shape())
    if(len(label.get_shape())==3):
        label=tf.squeeze(label,axis=0)
    
    # loss of classification
    #loss_cla=tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=label,
    #                                            pos_weight=tf.constant([1.0,1.0],dtype=tf.float32))
    if ( not flags.paras.isFocalLoss ) and flags.paras.isWeightedLoss:
        #loss_cla=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label)
        print('isWeightedLoss !')
        loss_cla=tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=label,
                                                pos_weight=tf.constant([4.0,1.0],dtype=tf.float32))
    elif( not flags.paras.isFocalLoss ) and (not flags.paras.isWeightedLoss):
        print('isSoftmaxLoss !')
        loss_cla=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label)
    else:
        print('isFocalLoss !')
        logits=tf.nn.softmax(logits)
        loss_cla=-label*tf.pow(1.0-logits,2.0)*tf.log(logits)-(1-label)*tf.pow(logits,2.0)*tf.log(1-logits)
    loss_cla=tf.reduce_mean(loss_cla, name='loss_cla')

    # tensorboard summary
    tf.summary.scalar('loss',loss_cla)

    ### SUM DEPTH LOSS
    loss_depth = 0.0
    maps_list = tf.split(maps, num_or_size_splits=len_seq, axis=-1)
    #assert(len(logits_list) == len(maps_list) - 1)
    for i in range(len(maps_list)-1):
        logits_map = logits_list[i]
        # loss of depth map 
        maps_reg = maps_list[i] / 255.0
        loss_depth_1= tf.pow(logits_map - maps_reg, 2)
        loss_depth_1 = tf.reduce_mean(loss_depth_1)
        # loss of contrast depth loss
        loss_depth_2 = util_network.contrast_depth_loss(logits_map, maps_reg)
        # total loss of depth 
        loss_depth_this = loss_depth_1 + loss_depth_2
        loss_depth += loss_depth_this

    loss_depth = loss_depth/float(len_seq)

    # loss of regularizer
    loss_reg=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total loss
    #loss = loss_cla + loss_reg + loss_depth
    #loss = loss_reg + loss_depth
    cla_ratio = flags.paras.cla_ratio
    depth_ratio = 1 - cla_ratio
    loss = loss_reg + depth_ratio * loss_depth +  cla_ratio * loss_cla

    correction=tf.equal(tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),\
                        tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32))
    accuracy=tf.reduce_mean(tf.cast(correction,dtype=tf.float32), name='accuracy')

    acc=tf.metrics.accuracy(labels=tf.cast(tf.argmax(label,axis=-1),dtype=tf.float32),
                             predictions=tf.cast(tf.argmax(logits,axis=-1),dtype=tf.float32) )
    eval_metric_ops = {"accuracy_metric": acc}
    tf.summary.scalar('accuracy', accuracy)

    return loss, loss_cla, loss_depth, accuracy, eval_metric_ops, logits_cla