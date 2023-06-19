import numpy as np
from Bio import SeqIO
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D,BatchNormalization, \
    Dropout, Dense, Activation, Concatenate, Multiply, GlobalMaxPooling1D, Add, Flatten, LSTM, Bidirectional, Layer,GRU
from keras.regularizers import l2
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

def read_fasta(fasta_file_name):
    seqs = []
    for fa in SeqIO.parse(fasta_file_name,"fasta"):
        seq = fa.seq
        if len(seq) < 201:
            continue
        else:
            seq1 = str(seq)[0:201]
        seqs.append(seq1)
    return seqs

def to_C2(seqs):
    base_dict = {
        'A': [0, 0],
        'C': [1, 1],
        'G': [1, 0],
        'U': [0, 1],
    }

    C2_seqs = []
    for seq in seqs:

        C2_matrix = np.zeros([len(seq), 2], dtype=float)
        index = 0
        for seq_base in seq:
            C2_matrix[index, base_dict[seq_base]] = 1
            index = index + 1

        C2_seqs.append(C2_matrix)
    return C2_seqs

def to_NCP(seqs):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0,], 'U': [0, 0, 1]}
    NCP_seqs = []
    for seq in seqs:
        NCP_matrix = np.zeros([len(seq),3],dtype=float)
        m = 0
        for seq_base in seq:
            NCP_matrix[m, :] = bases[seq_base]
            m += 1
        NCP_seqs.append(NCP_matrix)
    return NCP_seqs

def to_ND(seqs):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0,], 'U': [0, 0, 1]}
    ND_seq = []
    for seq in seqs:
        ND_matrix = np.zeros([len(seq), 4], dtype=float)
        cba = cbc = cbt = cbg =cball= 0
        for seq_base in seq:
            if seq_base == "A":
                cball += 1
                cba += 1
                Di = cba/cball
                ND_matrix[cball-1, :] = bases[seq_base] + [Di]
            elif seq_base == "C":
                cball += 1
                cbc += 1
                Di = cbc/cball
                ND_matrix[cball-1, :] = bases[seq_base] + [Di]
            elif seq_base == "G":
                cball += 1
                cbg += 1
                Di = cbg/cball
                ND_matrix[cball-1, :] = bases[seq_base] + [Di]
            elif seq_base == "U":
                cball += 1
                cbt += 1
                Di = cbt/cball
                ND_matrix[cball-1, :] = bases[seq_base] + [Di]
        ND_seq.append(ND_matrix)
    return ND_seq

def show_performance(y_true, y_pred):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    Sn = TP / (TP + FN + 1e-06)
    Sp = TN / (FP + TN + 1e-06)
    Acc = (TP + TN) / len(y_true)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))

def conv_factory(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('relu')(x)
    x = Conv1D(filters=filters,
               kernel_size=3,
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    return x

def transition(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def denseblock(x, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    for i in range(layers):
        x = conv_factory(x, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=-1)(list_feature_map)
        filters = filters + growth_rate
    return x, filters

def channel_attention(x, out_dim, ratio=16):

    shared_layer_one = Dense(units=out_dim // ratio, activation='relu',
                             use_bias=False, kernel_initializer='he_normal')
    shared_layer_two = Dense(units=out_dim, activation='sigmoid',
                             kernel_initializer='he_normal')

    avg_pool = GlobalAveragePooling1D()(x)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(x)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    pool = Add()([avg_pool, max_pool])
    scale = Multiply()([x, pool])

    return scale

def build_model(windows=201, denseblocks=4, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.5, weight_decay=1e-4):
    input_1 = Input(shape=(windows, 6))

    x_1 = Conv1D(filters=96, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_1)

    # Add denseblocks
    filters_1 = filters
    for i in range(denseblocks - 1):
        # Add denseblock
        x_1, filters_1 = denseblock(x_1, layers=layers,
                                    filters=filters_1, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)

        x_1 = BatchNormalization(axis=-1)(x_1)

        # Add transition
        x_1 = transition(x_1, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock
    # Add denseblock
    x_1, filters_1 = denseblock(x_1, layers=layers,
                                filters=filters_1, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    x_1 = BatchNormalization(axis=-1)(x_1)

    x_1 = channel_attention(x_1, filters_1, 16)

    x_2 = Bidirectional(LSTM(240, return_sequences=True))(x_1)
    x_2 = Dropout(0.5)(x_2)

    x_2 = channel_attention(x_2, filters_1, 16)

    x = GlobalMaxPooling1D()(x_2)

    x = Flatten()(x)

    x = Dense(units=240, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)

    x = Dense(units=40, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.2)(x)

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="DLC-ac4C")

    optimizer = Adam(lr=1e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':

    np.random.seed(6)
    tf.random.set_seed(6)

    WINDOWS =201

    train_seqs = read_fasta('../data/trainset.txt')
    train1 = np.array(to_C2(train_seqs)).astype(np.float32)
    train2 = np.array(to_ND(train_seqs)).astype(np.float32)
    train = np.concatenate([train1, train2], axis=2)

    train_label = np.array([1] * 2206 + [0] * 2206).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

    test_seqs = np.array(read_fasta('../data/testset.txt'))
    test1 = np.array(to_C2(test_seqs)).astype(np.float32)
    test2 = np.array(to_ND(test_seqs)).astype(np.float32)
    test = np.concatenate([test1, test2], axis=2)

    test_label = np.array([1] * 552+ [0] * 552).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)

    n = 10
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_performance = []

    for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
        print('*' * 30 + ' fold ' + str(fold_count+1) + ' ' + '*' * 30)
        trains, val = train[train_index], train[val_index]
        trains_label, val_label = train_label[train_index], train_label[val_index]

        BATCH_SIZE = 30
        EPOCHS = 300

        model1 = build_model()
        model2 = build_model()
        model3 = build_model()
        model4 = build_model()
        model5 = build_model()

        model1.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                   verbose=1)
        model1.save('../model/ac4c_model_' + str(fold_count + 1) + '_1.h5')
        del model1

        model2.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                   verbose=1)
        model2.save('../model/ac4c_model_' + str(fold_count + 1) + '_2.h5')
        del model2

        model3.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                   verbose=1)
        model3.save('../model/ac4c_model_' + str(fold_count + 1) + '_3.h5')
        del model3

        model4.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                   verbose=1)

        model4.save('../model/ac4c_model_' + str(fold_count + 1) + '_4.h5')
        del model4

        model5.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                   batch_size=BATCH_SIZE, shuffle=True,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                   verbose=1)
        model5.save('../model/ac4c_model_' + str(fold_count + 1) + '_5.h5')
        del model5

        model1 = load_model('../model/ac4c_model_' + str(fold_count + 1) + '_1.h5')
        model2 = load_model('../model/ac4c_model_' + str(fold_count + 1) + '_2.h5')
        model3 = load_model('../model/ac4c_model_' + str(fold_count + 1) + '_3.h5')
        model4 = load_model('../model/ac4c_model_' + str(fold_count + 1) + '_4.h5')
        model5 = load_model('../model/ac4c_model_' + str(fold_count + 1) + '_5.h5')

        val_score1 = model1.predict(val)
        val_score2 = model2.predict(val)
        val_score3 = model3.predict(val)
        val_score4 = model4.predict(val)
        val_score5 = model5.predict(val)
        all_score = val_score1 + val_score2 + val_score3 + val_score4 + val_score5
        val_score = all_score / 5

        Sn, Sp, Acc, MCC = show_performance(val_label[:, 1], val_score[:, 1])
        AUC = roc_auc_score(val_label[:, 1], val_score[:, 1])
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))

        performance = [Sn, Sp, Acc, MCC, AUC]
        all_performance.append(performance)

        '''Mapping the ROC'''
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
        fpr, tpr, thresholds = roc_curve(val_label[:, 1], val_score[:, 1], pos_label=1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC fold {} (AUC={:.4f})'.format(str(fold_count + 1), AUC))

        fold_count += 1
    all_performance = np.array(all_performance)
    print('10 fold result:',all_performance)
    performance_mean = performance_mean(all_performance)

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(np.array(all_performance)[:, 4])

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/10fold_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()


    model1 = build_model()
    model2 = build_model()
    model3 = build_model()
    model4 = build_model()
    model5 = build_model()

    model1.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True,
               callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
               verbose=0)
    model1.save('../model/model1.h5')
    del model1

    model2.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True,
               callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
               verbose=0)
    model2.save('../model/model2.h5')
    del model2

    model3.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True,
               callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
               verbose=0)
    model3.save('../model/model3.h5')
    del model3

    model4.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True,
               callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
               verbose=0)
    model4.save('../model/model4.h5')
    del model4

    model5.fit(x=train, y=train_label, validation_data=(test, test_label), epochs=EPOCHS,
               batch_size=BATCH_SIZE, shuffle=True,
               callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
               verbose=0)
    model5.save('../model/model5.h5')
    del model5

    model1 = load_model('../model/model1.h5')
    model2 = load_model('../model/model2.h5')
    model3 = load_model('../model/model3.h5')
    model4 = load_model('../model/model4.h5')
    model5 = load_model('../model/model5.h5')

    test_score1 = model1.predict(test)
    test_score2 = model2.predict(test)
    test_score3 = model3.predict(test)
    test_score4 = model4.predict(test)
    test_score5 = model5.predict(test)

    all_score = test_score1 + test_score2 + test_score3 + test_score4 + test_score5
    test_score = all_score / 5

    Sn, Sp, Acc, MCC = show_performance(test_label[:, 1], test_score[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_score[:, 1])

    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))
    '''Mapping the test ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    test_fpr, test_tpr, thresholds = roc_curve(test_label[:, 1], test_score[:, 1], pos_label=1)

    plt.plot(test_fpr, test_tpr, color='b', label=r'ac4c (AUC=%0.4f)' % (AUC), lw=2, alpha=.8)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/test_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()
