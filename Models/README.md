1. Perceptron

    简单的PLA，具体的算法参考了林轩田的PLA课程

2. LinearRegression

    线性回归，直接使用了normal equation

3. LogisticRegression

    逻辑回归，实现了批量梯度下降和随机梯度下降。

4. Adaboost

    只实现了分类问题，且特征只有数值型特征，弱的学习算法使用Decision Stump。实现参考了机器学习实战
    
5. DecisionTree

    使用ID3和C4.5构造树，本来想使用字典，但是觉得用类更好一些（因为有使用C结构体搭建树的经验），但是感觉挺麻烦的，尤其是搭建树时的递归调用。而且为了使Node节点适配ID3和C4.5和CART，所以在Node中有冗余属性哈。需要注意的是ID3和C4.5都是分类属性，而CART只有连续属性(因为CART是二叉树，对离散值的处理有点麻烦，就算了吧)。画出注解树就先饶了我吧。
    
6. PCA

7. Kmeans

    Kmeans随机选择的样本的影响，所以有二分Kmeans还有LBG等等依托于Kmeans的算法，只是简单的扩展，就不实现了。

8. kNN

9. GDBT

    
