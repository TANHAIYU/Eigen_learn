#include <iostream>
#include <ctime>
#include <Eigen/Dense>              // 稠密矩阵的代数运算
using namespace std;
#define MATRIX_SIZE 100
 
int main()
{
	// Eigen 以矩阵为基本数据单元，它是一个模板类，它的前三个参数为: 数据类型，行，列
	Eigen::Matrix<float, 2, 3> matrix_23;          // 声明一个2*3的float矩阵
	// 同时Eigen通过typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
	// 例如Vector3d实质上是Eigen::Matrix<double,3,1>,即三维向量
	Eigen::Vector3d v_3d;
	// 这是一样的
	Eigen::Matrix<float, 3, 1> vd_3d;
 
	// Matrix3d 实质上是Eigen::Matrix<double,3,3>
	Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();    // 初始化为零
 
	// 如果不确定矩阵大小，可以使用动态大小的矩阵
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
	// 下面是一种更简单的写法
	Eigen::MatrixXd matrix_x;
 
	// 下面是对Eigen矩阵的操作
	// 输入数据(初始化）
	matrix_23 << 1, 2, 3, 4, 5, 6;            
	// 输出
	cout << matrix_23 << endl;           
 
	// 用()访问矩阵中的元素
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			cout << matrix_23(i, j) << "\t";
	cout << endl;
 
	// 矩阵和向量的相乘(实际上仍然是矩阵和矩阵)
	v_3d << 3, 2, 1;
	vd_3d << 4, 5, 6;
 
	// 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的,下面是double和float
	// Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
	// 应该显式转换
	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << result << endl;
 
	// 标准的矩阵乘法
	Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
	cout << result2 << endl;
 
	// 矩阵的维数不对，会报错
	// Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
 
	// 一些矩阵运算
	// 四则运算就不演示了，直接用+-*/即可。
	matrix_33 = Eigen::Matrix3d::Random();      // 随机数矩阵
	cout << matrix_33 << endl;                  // 输出矩阵
	cout << "-------------------------" << endl;
	cout << matrix_33.transpose() << endl;      // 矩阵转置
	cout << "-------------------------" << endl;
	cout << matrix_33.sum() << endl;            // 各元素的和
	cout << "-------------------------" << endl;
	cout << matrix_33.trace() << endl;          // 矩阵的迹
	cout << "-------------------------" << endl;
	cout << 10 * matrix_33 << endl;             // 数乘
	cout << "-------------------------" << endl;
	cout << matrix_33.inverse() << endl;        // 矩阵求逆
	cout << "-------------------------" << endl;
	cout << matrix_33.determinant() << endl;    // 行列式
 
	// 特征值
	// 实对称矩阵可以保证对角化成功
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
	cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;   // 特征值对应的特征向量列排列'

	Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;           // 声明一个MATRIX_SIZE*MATRIX_SIZE矩阵
 
	matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);         // 矩阵初始化
 
	Eigen::Matrix< double, MATRIX_SIZE, 1> v_Nd;
	v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
 
	clock_t time_stt = clock();  //将当前时间赋值给time_stt里面
	// 直接求逆
	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse()*v_Nd;
	cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
 
	// 通常用矩阵分解来求，例如QR分解，速度会快很多
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);   // QR分解
	cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
	return 0;
}

 
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  Matrix3d m = Matrix3d::Random();//"Matrix3d"直接定义了一个3-by-3的方块矩阵
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3); //"Vector3d"直接定义了一个维度为3的列向量
  cout << "m * v =" << endl << m * v << endl;
}


 

固定尺寸与动态尺寸
什么时候应该使用固定尺寸（例如Matrix4f），什么时候应该使用动态尺寸（例如MatrixXf）？
简单的答案是：在可能的地方使用固定尺寸来显示非常小的尺寸，在需要的地方使用动态尺寸来显示较大的尺寸。
对于小尺寸，尤其是对于小于（大约）16的尺寸，使用固定尺寸对性能有极大的好处，因为它使Eigen避免了动态内存分配并展开了循环。在内部，固定大小的本征矩阵只是一个简单的数组，即Matrix4f mymatrix;真的等于只是在做float[16]; 因此这确实具有零运行时间成本。相比之下，动态大小矩阵的数组始终分配在堆上，因此MatrixXf mymatrix（行，列）;等于做float * mymatrix = new [行*列];除此之外，MatrixXf对象还将其行数和列数存储为成员变量。当然，使用固定大小的限制是，只有当您在编译时知道大小时，才有可能这样做。同样，对于足够大的尺寸（例如，对于大于（大约）32的尺寸），使用固定尺寸的性能优势变得可以忽略不计。更糟糕的是，尝试使用函数内部的固定大小创建非常大的矩阵可能会导致堆栈溢出，因为Eigen会尝试自动将数组分配为局部变量，而这通常是在堆栈上完成的。最后，视情况而定，当使用动态尺寸时，Eigen还可尝试进行矢量化（使用SIMD指令），请参见参考矢量化。

MatrixXcf a = MatrixXcf::Random(2, 2); //MatrixXcf 为复数矩阵
                cout << "Here is the matrix a\n"
                     << a << endl; //本身
                cout << "Here is the matrix a^T\n"
                     << a.transpose() << endl;    //转置
                cout << "Here is the conjugate of a\n"
                     << a.conjugate() << endl;    //共轭
                cout << "Here is the matrix a^*\n"
                     << a.adjoint() << endl;       //伴随



// 注意：对于一个矩阵自身的转置，应该使用.transposeInPlace()
        {
                Matrix2i a;
                a << 1, 2, 3, 4;
                cout << "Here is the matrix a:\n"
                     << a << endl;
                // 坑～～～～～～不应该这样～～～～～～～～
                // a = a.transpose(); // !!! do NOT do this !!!
                // cout << "and the result of the aliasing effect:\n"
                //      << a << endl;

                // 应该这样～～～～
                a.transposeInPlace();
                cout << "and after being transposed:\n"
                     << a << endl;
        }


mat << 1, 2,3, 4;
        //元素和，元素乘积，元素均值，最小系数，最大系数，迹
        cout << "Here is mat.sum():       " << mat.sum() << endl;
        cout << "Here is mat.prod():      " << mat.prod() << endl;
        cout << "Here is mat.mean():      " << mat.mean() << endl;
        cout << "Here is mat.minCoeff():  " << mat.minCoeff() << endl;
        cout << "Here is mat.maxCoeff():  " << mat.maxCoeff() << endl;
        cout << "Here is mat.trace():     " << mat.trace() << endl;

// 可以返回元素位置
        Matrix3f m = Matrix3f::Random();
        std::ptrdiff_t i, j;
        float minOfM = m.minCoeff(&i, &j);
        cout << "Here is the matrix m:\n"
             << m << endl;
        cout << "Its minimum coefficient (" << minOfM
             << ") is at position (" << i << "," << j << ")\n\n";
        RowVector4i v = RowVector4i::Random();
        int maxOfV = v.maxCoeff(&i);
        cout << "Here is the vector v: " << v << endl;
        cout << "Its maximum coefficient (" << maxOfV
             << ") is at position " << i << endl;


        ArrayXXf m(2, 2); // 二维动态float类型数组

        // assign some values coefficient by coefficient
        m(0, 0) = 1.0;
        m(0, 1) = 2.0;
        m(1, 0) = 3.0;
        m(1, 1) = m(0, 1) + m(1, 0);

        // print values to standard output
        cout << m << endl
             << endl;

        // using the comma-initializer is also allowed
        m << 1.0, 2.0, 3.0, 4.0;

        // print values to standard output
        cout << m << endl;


//您可以将一个数组乘以标量，这与矩阵的工作方式相同。
        //数组与矩阵根本不同的地方是将两个矩阵相乘。
        //矩阵将乘法解释为矩阵乘积，而数组将乘法解释为按系数乘积。
        //因此，当且仅当两个数组具有相同的维数时，它们才能相乘
        LOG();
        ArrayXXf a(2, 2);
        ArrayXXf b(2, 2);
        a << 1, 2,
            3, 4;
        b << 5, 6,
            7, 8;
        cout << "a * b = " << endl
             << a * b << endl;
        // Output is:
        // a * b =
        //  5 12
        // 21 32
}

什么时候应该使用Matrix类的对象，什么时候应该使用Array类的对象？
您不能对数组应用矩阵运算，也不能对矩阵应用数组运算。因此，如果您需要进行线性代数运算（例如矩阵乘法），则应使用矩阵。如果需要进行系数运算，则应使用数组。但是，有时并不是那么简单，但是您需要同时使用Matrix和Array操作。在这种情况下，您需要将矩阵转换为数组或反向转换。无论选择将对象声明为数组还是矩阵，都可以访问所有操作。矩阵表达式具有.array()方法，可以将它们"转换"为数组表达式，因此可以轻松地应用按系数进行运算。相反，数组表达式具有.matrix()方法。与所有Eigen表达式抽象一样，这没有任何运行时开销（只要您让编译器进行优化）.array（）和.matrix() 可被用作右值和作为左值。Eigen禁止在表达式中混合矩阵和数组。例如，您不能直接矩阵和数组相加。运算符+的操作数要么都是矩阵，要么都是数组。但是，使用.array（）和.matrix（）可以轻松地将其转换为另一种。
 //～～～～注意～～～～～
此规则的例外是赋值运算符=：允许将矩阵表达式分配给数组变量，或将数组表达式分配给矩阵变量。下面的示例演示如何通过使用.array（）方法对Matrix对象使用数组操作。例如，语句需要两个矩阵和，他们两个转换为阵列，用来将它们相乘系数明智并将结果指定给矩阵变量（这是合法的，因为本征允许分配数组表达式到矩阵的变量）。result = m.array() * n.array()mnresult
实际上，这种使用情况非常普遍，以至于Eigen为矩阵提供了const .cwiseProduct()方法来计算系数乘积。
        MatrixXf m(2, 2);
        MatrixXf n(2, 2);
        MatrixXf result(2, 2);
        m << 1, 2,
            3, 4;
        n << 5, 6,
            7, 8;
        result = m * n;
        cout << "-- Matrix m*n: --" << endl
             << result << endl
             << endl;
        result = m.array() * n.array();
        cout << "-- Array m*n: --" << endl
             << result << endl
             << endl;
        result = m.cwiseProduct(n);
        cout << "-- With cwiseProduct: --" << endl
             << result << endl
             << endl;
        result = m.array() + 4;
        cout << "-- Array m + 4: --" << endl
             << result << endl
             << endl;
        // Output is:
        // -- Matrix m*n: --
        // 19 22
        // 43 50
        // -- Array m*n: --
        //  5 12
        // 21 32
        // -- With cwiseProduct: --
        //  5 12
        // 21 32
        // -- Array m + 4: --
        // 5 6
        // 7 8

切片
Eigen::Matrix4f m;
        m << 1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16;
        cout << "m.leftCols(2) =" << endl
             << m.leftCols(2) << endl
             << endl;
        cout << "m.bottomRows<2>() =" << endl
             << m.bottomRows<2>() << endl
             << endl;
        m.topLeftCorner(1, 3) = m.bottomRightCorner(3, 1).transpose();
        cout << "After assignment, m = " << endl
             << m << endl;
        // Output is:
        // m.leftCols(2) =
        //  1  2
        //  5  6
        //  9 10
        // 13 14
        // m.bottomRows<2>() =
        //  9 10 11 12
        // 13 14 15 16
        // After assignment, m =
        //  8 12 16  4
        //  5  6  7  8
        //  9 10 11 12
        // 13 14 15 16

Eigen::ArrayXf v(6);
        v << 1, 2, 3, 4, 5, 6;
        cout << "v.head(3) =" << endl
             << v.head(3) << endl
             << endl;
        cout << "v.tail<3>() = " << endl
             << v.tail<3>() << endl
             << endl;
        v.segment(1, 4) *= 2;
        cout << "after 'v.segment(1,4) *= 2', v =" << endl
             << v << endl;

        // Output is:
        // v.head(3) =
        // 1
        // 2
        // 3

        // v.tail<3>() =
        // 4
        // 5
        // 6

        // after 'v.segment(1,4) *= 2', v =
        //  1
        //  4
        //  6
        //  8
        // 10
        //  6

//此外，初始化列表的元素本身可以是向量或矩阵。通常的用途是将向量或矩阵连接在一起。例如，这是将两个行向量连接在一起的方法。注意：～～～～～请记住，必须先设置大小，然后才能使用逗号初始化程序
        RowVectorXd vec1(3);
        vec1 << 1, 2, 3;
        RowVectorXd vec2(4);
        vec2 << 1, 4, 9, 16;
        RowVectorXd joined(7);
        joined << vec1, vec2;

        //  对矩阵的某一块赋值
        Matrix3f m;
        m.row(0) << 1, 2, 3;
        m.block(1, 0, 2, 2) << 4, 5, 7, 8;
        m.col(2).tail(2) << 6, 9;

        特殊矩阵的表示方法
// 模板类Matrix<>和Array<>有静态方法，可以帮助初始化；
        //有三种变体:
        //第一个变体不带参数，只能用于固定大小的对象。如果要将动态尺寸对象初始化为零，则需要指定尺寸。
        //第二个变体需要一个参数，并且可以用于一维动态尺寸对象，
        //第三个变体需要两个参数，并且可以用于二维对象。

        std::cout << "A fixed-size array:\n";
        Array33f a1 = Array33f::Zero();   //创建float3*3的矩阵，并且将值全部设为0
        std::cout << "A one-dimensional dynamic-size array:\n";
        ArrayXf a2 = ArrayXf::Zero(3);    //创建动态向量，并且将值设为0
        std::cout << "A two-dimensional dynamic-size array:\n";
        ArrayXXf a3 = ArrayXXf::Zero(3, 4); //创建动态矩阵，并且将值设为0

        //同样，静态方法Constant(value)会将所有系数设置为value。
        // 如果需要指定对象的大小，则附加参数放在value参数之前，如
        // MatrixXd::Constant(rows, cols, value)。
        //Identity()获得单位矩阵, 此方法仅适用于Matrix，不适用于Array，因为"单位矩阵"是线性代数概念。
        //该方法LinSpaced（尺寸，低，高）是仅可用于载体和一维数组; 它产生一个指定大小的向量，其系数在low和之间平均间隔high。
        //方法LinSpaced()以下示例说明了该示例，该示例打印一张表格，其中包含以度为单位的角度，以弧度为单位的相应角度以及它们的正弦和余弦值。
        ArrayXXf table(10, 4);  //初始化为
        table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
        table.col(1) = M_PI / 180 * table.col(0);
        table.col(2) = table.col(1).sin();
        table.col(3) = table.col(1).cos();

        //Eigen定义了诸如setZero()，MatrixBase :: setIdentity（）和DenseBase :: setLinSpaced()之类的实用程序函数来方便地执行此操作。
        //即，可以采用对象的成员函数设置初始值。
        //下面的示例对比了三种构造矩阵的J =[O  I ; I O ] 方法
        // 使用静态方法和operator=
        const int size = 6;
        MatrixXd mat1(size, size);
        mat1.topLeftCorner(size / 2, size / 2) = MatrixXd::Zero(size / 2, size / 2);
        mat1.topRightCorner(size / 2, size / 2) = MatrixXd::Identity(size / 2, size / 2);
        mat1.bottomLeftCorner(size / 2, size / 2) = MatrixXd::Identity(size / 2, size / 2);
        mat1.bottomRightCorner(size / 2, size / 2) = MatrixXd::Zero(size / 2, size / 2);

        //使用.setXxx()方法
        MatrixXd mat2(size, size);
        mat2.topLeftCorner(size / 2, size / 2).setZero();
        mat2.topRightCorner(size / 2, size / 2).setIdentity();
        mat2.bottomLeftCorner(size / 2, size / 2).setIdentity();
        mat2.bottomRightCorner(size / 2, size / 2).setZero();

        MatrixXd mat3(size, size);
        //使用静态方法和逗号初始化
        mat3 << MatrixXd::Zero(size / 2, size / 2), MatrixXd::Identity(size / 2, size / 2),
            MatrixXd::Identity(size / 2, size / 2), MatrixXd::Zero(size / 2, size / 2);

void UsageAsTemporaryObjects()  //使用静态对象
{
        //如上所示，可以在声明时或在赋值运算符的右侧使用静态方法Zero()和Constant()来初始化变量。您可以将这些方法视为返回矩阵或数组。实际上，它们返回所谓的**表达式对象**，这些表达式对象在需要时求值到矩阵或数组，因此该语法不会产生任何开销。这些表达式也可以用作临时对象。
        MatrixXd m = MatrixXd::Random(3, 3);
        m = (m + MatrixXd::Constant(3, 3, 1.2)) * 50;  //使用constant初始化变量
        v << 1, 2, 3;

        // The comma-initializer, too, can also be used to construct temporary objects. The following example constructs a random matrix of size 2-by-3,and then multiplies this matrix on the left with [0 1; 1 0] 逗号初始化程序也可以用于构造临时对象。 下面的示例构造一个大小为2×3的随机矩阵，然后将该左边的矩阵乘以[0 1; 1 0]
        MatrixXf mat = MatrixXf::Random(2, 3);   //随机生成2*3的浮点数矩阵
        // 在完成临时子矩阵的逗号初始化之后，必须使用finish（）方法来获取实际的矩阵对象。finished 类似于endl，让它立即生成
        mat = (MatrixXf(2, 2) << 0, 1, 1, 0).finished() * mat;
        std::cout << mat << std::endl;
}
void NormComputations()
{
        // 这些运算也可以在矩阵上运算。在这种情况下，n×p矩阵被视为大小（n * p）的向量，
        // 因此，例如norm()方法返回" Frobenius"或" Hilbert-Schmidt"范数。
        // 如果需要其他按系数分配的l^p范数，请使用lpNorm <p>()。
        // 如果需要无穷范数，则模板参数p可以采用特殊值Infinity，这是系数绝对值的最大值。
        VectorXf v(2);
        MatrixXf m(2, 2), n(2, 2);
        v << -1,
            2;

        m << 1, -2,
            -3, 4;
        // 向量范数
        cout << "v.squaredNorm() = " << v.squaredNorm() << endl;            //2范数
        cout << "v.norm() = " << v.norm() << endl;                           //返回.squaredNorm()的平方根
        cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << endl;                //一阶范数
        cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Infinity>() << endl;       //无穷范数
        cout << endl;
        // 矩阵范数
        cout << "m.squaredNorm() = " << m.squaredNorm() << endl;
        cout << "m.norm() = " << m.norm() << endl;
        cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << endl;
        cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Infinity>() << endl;

        // 也可自己计算1范数和无穷范数
        MatrixXf mat(2, 2);
        mat << 1, -2,
            -3, 4;
        cout << "1-norm(mat)     = " << mat.cwiseAbs().colwise().sum().maxCoeff()
             << " == " << mat.colwise().lpNorm<1>().maxCoeff() << endl;
        cout << "infty-norm(mat) = " << mat.cwiseAbs().rowwise().sum().maxCoeff()
             << " == " << mat.rowwise().lpNorm<1>().maxCoeff() << endl;
}

数据归约是指在尽可能保持数据原貌的前提下，最大限度地精简数据量（完成该任务的必要前提是理解挖掘任务和熟悉数据本身内容）Mat.colwise()理解为分别去看矩阵的每一列，然后再作用maxCoeff()函数，即求每一列的最大值。需要注意的是，colwise返回的是一个行向量（列方向降维），rowwise返回的是一个列向量（行方向降维）。

Aliasing
    混淆往往出现在赋值号左右都用同一变量的时候，如mat = mat.transpose();解决方法之一是使用eval()函数，看下面两个实例
  
    如果有xxxInplace()函数，用这个函数可达到最高效的效果，如transposeInPlace()
 
    通常情况下，如果元素的表达只与该元素自己有关，那么一般不会引起混淆，如语句：mat = (2 * mat - MatrixXf::Identity(2, 2)).array().square();

void Visitors()
{
        // 这是在矩阵和数组的所有元素中
        //想要获得一个系数在Matrix或Array中的位置时，访问者很有用。
        //最简单的示例是maxCoeff（＆x，＆y）和minCoeff（＆x，＆y），可用于查找Matrix或Array中最大或最小系数的位置。
        //传递给访问者的参数是指向要存储行和列位置的变量的指针。这些变量应为Index类型，

        Eigen::MatrixXf m(2, 2);

        m << 1, 2,
            3, 4;
        //得到最大值的位置
        MatrixXf::Index maxRow, maxCol;
        float max = m.maxCoeff(&maxRow, &maxCol);
        //得到最小值的位置
        MatrixXf::Index minRow, minCol;
        float min = m.minCoeff(&minRow, &minCol);
}

void PartialReductions() 部分缩减
{
        // 记住,element-wise是按元素的，那么colwise()或rowwise()表示按列或行的
        //部分归约是可以在Matrix或Array上按列或按行操作的归约，对每个列或行应用归约运算并返回具有相应值的列或行向量。
        //一个简单的示例是获取给定矩阵中每一列中元素的最大值，并将结果存储在行向量中：
        // column-wise返回行向量，row-wise返回列向量。啥意思？应该设计底层操作，以后再看   
        Eigen::MatrixXf mat(2, 4);
        mat << 1, 2, 6, 9,
            3, 1, 7, 2;
        std::cout << "Column's maximum: " << std::endl
                  << mat.colwise().maxCoeff() << std::endl; // 对于矩阵mat的每一列，取最大系数值
        // 也可以对行操作
        std::cout << "Row's maximum: " << std::endl
                  << mat.rowwise().maxCoeff() << std::endl; // 对于矩阵mat的每一行，取最大系数值
}
void CombiningPartialReductionsWithOtherOperations()
{
        MatrixXf mat(2, 4);
        mat << 1, 2, 6, 9,
            3, 1, 7, 2;

        MatrixXf::Index maxIndex;
        float maxNorm = mat.colwise().sum().maxCoeff(&maxIndex); //  对于矩阵的每一列中的元素求和，结果的最大系数在第2列
        std::cout << "Maximum sum at position " << maxIndex << std::endl;
        std::cout << "The corresponding vector is: " << std::endl;
        std::cout << mat.col(maxIndex) << std::endl;
        std::cout << "And its sum is is: " << maxNorm << std::endl;
}

void Broadcasting()广播机制
{
        LOG();
        //广播背后的概念类似于部分归约，区别在于广播构造了一个表达式，其中向量（列或行）通过在一个方向上复制而被解释为矩阵。
        //一个简单的示例是将某个列向量添加到矩阵中的每一列。这可以通过以下方式完成：
        Eigen::MatrixXf mat(2, 4);
        Eigen::VectorXf v(2);
        mat << 1, 2, 6, 9,
            3, 1, 7, 2;
        v << 0,            //列向量
            1;

        //add v to each column of m
        //mat.colwise() += v用两种等效的方式解释指令。
        //它将向量添加v到矩阵的每一列。或者，可以将其解释为重复向量v四次以形成四乘二矩阵，然后将其加到mat
        mat.colwise() += v;
        std::cout << "Broadcasting result: " << std::endl;
        std::cout << mat << std::endl;
        // Output is:
        // Broadcasting result:
        // 1 2 6 9
        // 4 2 8 3

        // 在矩阵上，我们可以执行-=,+=,+,-操作，但是不能进行*=,/=,*,/操作
        // 在数组上我们可执行*=,/=,*,/操作
        // If you want multiply column 0 of a matrix mat with v(0), column 1 with v(1), and so on, then use mat = mat * v.asDiagonal().要逐列或逐行添加的向量必须为Vector类型，并且不能为Matrix。如果不满足，则会出现编译时错误。广播操作只能应用于Vector类型的对象。这同样适用于数组类VectorXf是ArrayXf。

        // 同样，也可以对行执行此操作
        {
                Eigen::MatrixXf mat(2, 4);
                Eigen::VectorXf v(4);  
                mat << 1, 2, 6, 9,
                    3, 1, 7, 2;
                v << 0, 1, 2, 3;          //列向量
                //add v to each row of m
                mat.rowwise() += v.transpose();
                std::cout << "Broadcasting result: " << std::endl;
                std::cout << mat << std::endl;

                //  Broadcasting result:
                //  1  3  8 12
                //  3  2  9  5
        }
}

void CombiningBroadcastingWithOtherOperations()      //把广播机制与其他操作结合在一起
{
        // 广播还可以与其他操作（例如矩阵或阵列操作），归约和部分归约相结合。
        //现在已经介绍了广播，约简和部分约简，我们可以深入研究一个更高级的示例，该示例v在矩阵的列中找到向量的最近邻m。欧几里德距离将在本示例中使用，计算具有部分归约名为squaredNorm()的平方欧几里德距离：
        Eigen::MatrixXf m(2, 4);
        Eigen::VectorXf v(2);
        m << 1, 23, 6, 9,
            3, 11, 7, 2;
        v << 2,
            3;
        MatrixXf::Index index;
        // find nearest neighbour
        (m.colwise() - v).colwise().squaredNorm().minCoeff(&index);
        cout << "Nearest neighbour is column " << index << ":" << endl;
        cout << m.col(index) << endl;
}

MAP的使用
//有时您可能要在Eigen中使用预定义的数字数组(C++)作为矢量或矩阵(Eigen)。
//一种选择是复制数据，但最常见的情况是您可能希望将此内存。幸运的是，使用Map类非常容易。
// Map 类 实现C++中的数组内存和Eigen对象的交互
// Map< Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>  >  请注意，在这种默认情况下，Map仅需要一个模板参数。 要构造Map变量，您还需要其他两条信息：指向定义系数数组的内存区域的指针，以及所需的矩阵或矢量形状。(注意区分模板参数和函数形参)
// 例如，要定义一个float在编译时确定大小的矩阵，您可以执行以下操作：Map <MatrixXf> mf（pf，rows，columns）;
// 其中pf是一个float *指向的存储器阵列。固定大小的整数只读向量可能会声明为
// Map <const Vector4i> mi（pi）;
// 其中pi是int *。在这种情况下，不必将大小传递给构造函数，因为它已经由Matrix / Array类型指定。
// 请注意，Map没有默认的构造函数。您必须传递一个指针来初始化对象。但是，您可以解决此要求（请参阅更改Map数组）。
// Map足够灵活，可以容纳各种不同的数据表示形式。还有其他两个（可选）模板参数：
// Map<typename MatrixType, int MapOptions(MapOptions指定指针是Aligned还是Unaligned。默认值为Unaligned),  typename StrideType>
//StrideType允许您使用Stride类为内存阵列指定自定义布局。一个示例是指定数据数组以行优先格式进行组织MapConstruct()

void MapConstruct()
{
        int array[8];
        for (int i = 0; i < 8; ++i)
                array[i] = i;
        cout << "Column-major:\n"
             << Map<Matrix<int, 2, 4>>(array) << endl;    //以列为主
        cout << "Row-major:\n"
             << Map<Matrix<int, 2, 4, RowMajor>>(array) << endl; //以行为主
        cout << "Row-major using stride:\n"//
             << Map<Matrix<int, 2, 4>, Unaligned, Stride<1, 4>>(array) << endl;
        //Output is:
        //Column-major:
        // 0 2 4 6
        // 1 3 5 7
        // Row-major:
        // 0 1 2 3
        // 4 5 6 7
        // Row-major using stride:
        // 0 1 2 3
        // 4 5 6 7

        //但是，Stride比这更灵活。 有关详细信息，请参见Map和Stride类的文档。
}

void UsingMapVariables()
{

        typedef Matrix<float, 1, Dynamic> MatrixType;
        typedef Map<MatrixType> MapType;
        typedef Map<const MatrixType> MapTypeConst;              // a read-only map
        const int n_dims = 5;

        MatrixType m1(n_dims), m2(n_dims);
        m1.setRandom();                        //m1设为随机-1到1 
        m2.setRandom();
        float *p = &m2(0);                        // 获取存储m2数据的地址
        MapType m2map(p, m2.size());            // m2map与m2共享数据
        MapTypeConst m2mapconst(p, m2.size());  // 一个m2的只读访问器
        cout << "m1: " << m1 << endl;
        cout << "m2: " << m2 << endl;
        cout << "Squared euclidean distance: " << (m1 - m2).squaredNorm() << endl;
        cout << "Squared euclidean distance, using map: " << (m1 - m2map).squaredNorm() << endl;
        m2map(3) = 7;                       // this will change m2, since they share the same array
        cout << "Updated m2: " << m2 << endl;
        cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << endl;
        //Output is:
        // m1:   0.68 -0.211  0.566  0.597  0.823
        // m2: -0.605  -0.33  0.536 -0.444  0.108
        // Squared euclidean distance: 3.26
        // Squared euclidean distance, using map: 3.26
        // Updated m2: -0.605  -0.33  0.536      7  0.108
        // m2 coefficient 2, constant accessor: 0.536
}

void ChangingTheMappedArray()  //改变已经map的向量
{
        //可以使用C ++" placement new"(位置new，在程序员给定的内存放置元素) 语法更改已声明的Map对象的数组。尽管有出现，但它不会调用内存分配器，因为语法指定了存储结果的位置。简单的说，位置new只是在指定位置写入内容，并不会像new一样，先在堆上分配内存，然后再依次初始化对象，这也是为什么叫位置new，因为它会按照我们指定的位置构造对象
        int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        Map<RowVectorXi> v(data, 4);                     //只将前四个放进map
        cout << "The mapped vector v is: " << v << "\n";     
        new (&v) Map<RowVectorXi>(data + 4, 5);           //在指定位置写入内容
        cout << "Now v is: " << v << "\n";
        //The mapped vector v is: 1 2 3 4
        //Now v is: 5 6 7 8 9
        // TODO : 教程中没有指定n_matrices
        // VectorXf b(n_matrices);
        // for (int i = 0; i < n_matrices; i++)
        // {
        //         new (&A) Map<Matrix3f>(get_matrix_pointer(i));
        //         b(i) = A.trace();}
}
}
void Reshape() //重塑
{
        //Reshape操作在于修改矩阵的大小，同时保持相同的系数。除了修改输入矩阵本身（这对于编译时大小而言是不可能的）之外，该方法还包括使用Map类在存储上创建不同的视图。这是创建矩阵的一维线性视图的典型示例：
        MatrixXf M1(3, 3);                   // Column-major storage
        // 注意：逗号初始化是为了方便我们输入矩阵，但是底层存储是按照列主的顺序存储的
        M1 << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
        Map<RowVectorXf> v1(M1.data(), M1.size());
        cout << "v1:" << endl
             << v1 << endl;
        Matrix<float, Dynamic, Dynamic, RowMajor> M2(M1);//此处默认是ColMajor
        Map<RowVectorXf> v2(M2.data(), M2.size());
        cout << "v2:" << endl
             << v2 << endl;
        //Output is:
        // v1:
        // 1 4 7 2 5 8 3 6 9
        // v2:
        // 1 2 3 4 5 6 7 8 9
        //注意输入矩阵的存储顺序如何修改线性视图中系数的顺序。这是另一个将2x6矩阵重塑为6x2矩阵的示例：
        {
                MatrixXf M1(2, 6); // Column-major storage
                M1 << 1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12;
                Map<MatrixXf> M2(M1.data(), 6, 2);
                cout << "M2:" << endl
                     << M2 << endl;
                //Output is:
                //  M2:
                //  1  4
                //  7 10
                //  2  5
                //  8 11
                //  3  6
                //  9 12
        }
}

void Slicing()  //切片
{
        //切片包括获取一组在矩阵内均匀间隔的行，列或元素。再次，Map类可以轻松模仿此功能。
        //例如，可以跳过向量中的每个P元素：
        RowVectorXf v = RowVectorXf::LinSpaced(20, 0, 19);
        cout << "Input:" << endl
             << v << endl;
        Map<RowVectorXf, 0, InnerStride<2>> v2(v.data(), v.size() / 2); 
//此处的InnerStride<2>指的是内部步长是2
        cout << "Even:" << v2 << endl;
        
//Output is:
        //  Input:S
        //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
        // Even: 0  2  4  6  8 10 12 14 16 18

        // TODO:这个好像有点复杂，让我缓缓～
        MatrixXf M1 = MatrixXf::Random(3, 8);
        cout << "Column major input:" << endl
             << M1 << "\n";
        Map<MatrixXf, 0, OuterStride<>> M2(M1.data(), M1.rows(), (M1.cols() + 2) / 3, OuterStride<>(M1.outerStride() * 3));    //选中行进行操作，
        cout << "1 column over 3:" << endl
             << M2 << "\n";
        typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMajorMatrixXf;
        RowMajorMatrixXf M3(M1);
        cout << "Row major input:" << endl
             << M3 << "\n";
        Map<RowMajorMatrixXf, 0, Stride<Dynamic, 3>> M4(M3.data(), M3.rows(), (M3.cols() + 2) / 3,
                                                        Stride<Dynamic, 3>(M3.outerStride(), 3));
        cout << "1 column over 3:" << endl
             << M4 << "\n";
        //Column major input:
        //    0.68   0.597   -0.33   0.108   -0.27   0.832  -0.717  -0.514
        //  -0.211   0.823   0.536 -0.0452  0.0268   0.271   0.214  -0.726
        //   0.566  -0.605  -0.444   0.258   0.904   0.435  -0.967   0.608
        // 1 column over 3:
        //    0.68   0.108  -0.717
        //  -0.211 -0.0452   0.214
        //   0.566   0.258  -0.967
        // Row major input:
        //    0.68   0.597   -0.33   0.108   -0.27   0.832  -0.717  -0.514
        //  -0.211   0.823   0.536 -0.0452  0.0268   0.271   0.214  -0.726
        //   0.566  -0.605  -0.444   0.258   0.904   0.435  -0.967   0.608
        // 1 column over 3:
        //    0.68   0.108  -0.717
        //  -0.211 -0.0452   0.214
        //   0.566   0.258  -0.967
}



混淆
// 这一节比较绕
// 在Eigen中，混淆(aliasing)是指相同的矩阵（或数组或向量）出现在赋值运算符的左侧和右侧的赋值语句; 例如，A = AB ,  a = a^Tb  A=A*A。产生混淆的原因是Eigen采用惰性求值。混淆可能是有害的，也可能是无害的。有害的混淆，导致可能不正确的结果，无害的混淆可以产生正确的结果。

// 有害混淆
// 使用.eval()方法，可以解决混淆问题。具体的说，.eval()方法生成临时对象，然后再执行赋值操作。如果Eigen中假定该操作是混淆的，那么它会自动使用.eval()方法，而不需要我们显示调用。无害混淆是我们不需要评估，它对系数计算无害。这包括标量乘法和矩阵或数组加法。

// 将两个矩阵相乘时，Eigen会假定发生混叠(注意，Eigen3.3以后的版本中需要区分目标矩阵大小是否改变了)
//如果您知道没有混淆，则可以使用noalias()。

// 在所有其他情况下，Eigen假定不存在混叠问题，因此如果实际上发生混叠，则会给出错误的结果。
// 为防止这种情况，您必须使用eval（）或xxxInPlace（）函数之一。

void Aliasing()
{
        // 例如mat = 2 * mat(虽混淆但无害);或mat = mat.transpose();(有害的混淆)。

        MatrixXi mat(3, 3);
        mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        cout << "Here is the matrix mat:\n"
             << mat << endl;
        // This assignment shows the aliasing problem
        mat.bottomRightCorner(2, 2) = mat.topLeftCorner(2, 2);
        cout << "After the assignment, mat = \n"
             << mat << endl;

        // Output  is:
        // Here is the matrix mat:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        // After the assignment, mat =
        // 1 2 3
        // 4 1 2
        // 7 4 1

        // mat.bottomRightCorner（2,2）= mat.topLeftCorner（2,2）;
        // 该赋值具有混淆aliasing）：系数mat(1,1)既出现在mat.bottomRightCorner(2,2)分配左侧的块中mat.topLeftCorner(2,2)，又出现在右侧的块中。
        // 分配后，右下角的（2,2）项应具有mat(1,1)分配前的值5。但是，输出显示mat(2,2)实际上为1。
        // 问题在于Eigen使用了惰性求值。结果类似于
        // mat（1,1）= mat（0,0）;
        // mat（1,2）= mat（0,1）;
        // mat（2,1）= mat（1,0）;
        // mat（2,2）= mat（1,1）;
        // 因此，mat(2,2)分配了新值mat(1,1)而不是旧值。
        // 可以通过调用eval()解决此问题(注:eval()负责生成临时对象而避免混淆)
        // 尝试缩小矩阵时，混淆也会发生。
        // 例如，表达式vec = vec.head(n)和mat = mat.block(i,j,r,c)具有混淆。
        //通常，在编译时无法检测到混淆：如果mat在第一个示例中稍大一点，则块将不会重叠，也不会出现混淆问题。
        //但是Eigen确实会在运行时检测到一些混淆实例。Matrix和向量算术中提到了以下显示别名的示例：

        // Matrix2i a;
        // a << 1, 2, 3, 4;
        // cout << "Here is the matrix a:\n"
        //      << a << endl;
        // a = a.transpose(); // !!! do NOT do this !!!
        // cout << "and the result of the aliasing effect:\n"
        //      << a << endl;

        // 输出显示混淆(alising)问题。
        // 但是，默认情况下，Eigen使用运行时断言来检测到此情况并退出，并显示如下消息
        // void Eigen::DenseBase<Derived>::checkTransposeAliasing(const OtherDerived&) const
        // [with OtherDerived = Eigen::Transpose<Eigen::Matrix<int, 2, 2, 0, 2, 2> >, Derived = Eigen::Matrix<int, 2, 2, 0, 2, 2>]:
        // Assertion `(!internal::check_transpose_aliasing_selector<Scalar,internal::blas_traits<Derived>::IsTransposed,OtherDerived>::run(internal::extract_data(derived()), other))
        // && "aliasing detected during transposition, use transposeInPlace() or evaluate the rhs into a temporary using .eval()"' failed.

        //用户可以通过定义EIGEN_NO_DEBUG宏来关闭Eigen的运行时断言
}

void ResolvingAliasingIssues()  //解决混淆问题的办法=.eval()
{
        //解决方法：Eigen必须将右侧完全看作一个临时矩阵/数组，然后将其分配给左侧。
        //函数**eval()**正是这样做的,作用为生成一个临时对象
        MatrixXi mat(3, 3);
        mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        cout << "Here is the matrix mat:\n"
             << mat << endl;
        // The eval() solves the aliasing problem
        mat.bottomRightCorner(2, 2) = mat.topLeftCorner(2, 2).eval();
        cout << "After the assignment, mat = \n"
             << mat << endl;

        //相同的解决方案也适用于第二示例中，与转置：只需更换线a = a.transpose();用a = a.transpose().eval();。但是，在这种常见情况下，有更好的解决方案。
        //Eigen提供了专用函数transposeInPlace()，该函数通过其转置来替换矩阵。如下所示：
        MatrixXf a(2, 3);
        a << 1, 2, 3, 4, 5, 6;
        a.transposeInPlace();
             
        //如果xxxInPlace()函数可用，则最好使用它，因为它可以更清楚地指示您正在做什么。
        //这也可以让Eigen更积极地进行优化。这些是提供的一些xxxInPlace()函数：
        // Original function	                      In-place function
        // MatrixBase::adjoint()	          MatrixBase::adjointInPlace()
        // DenseBase::reverse()	                 DenseBase::reverseInPlace()
        // LDLT::solve()	                            LDLT::solveInPlace()
        // LLT::solve()	                                      LLT::solveInPlace()
        // TriangularView::solve()	        TriangularView::solveInPlace()
        // DenseBase::transpose()	       DenseBase::transposeInPlace()

        //在特殊情况下，矩阵或向量使用类似的表达式缩小vec = vec.head(n)，则可以使用conservativeResize()。
}

void AliasingAndComponentWiseOperations()
{
        //如果同一矩阵或数组同时出现在赋值运算符的左侧和右侧，则可能很危险，因此通常有必要显示地评估右侧. 但是，应用基于元素的操作（例如矩阵加法，标量乘法和数组乘法）是安全的。以下示例仅具有基于组件的操作。因此，即使相同的矩阵出现在赋值符号的两侧，也不需要eval()。
        MatrixXf mat(2, 2);
        mat << 1, 2, 4, 7;
        cout << "Here is the matrix mat:\n"
             << mat << endl
             << endl;
        mat = 2 * mat;
        cout << "After 'mat = 2 * mat', mat = \n"
             << mat << endl
             << endl;
        mat = mat - MatrixXf::Identity(2, 2);
        cout << "After the subtraction, it becomes\n"
             << mat << endl
             << endl;
        ArrayXXf arr = mat;
        arr = arr.square();
        cout << "After squaring, it becomes\n"
             << arr << endl
             << endl;
        // Combining all operations in one statement:
        mat << 1, 2, 4, 7;
        mat = (2 * mat - MatrixXf::Identity(2, 2)).array().square();
        cout << "Doing everything at once yields\n"
             << mat << endl
             << endl;
        // Output is:
        // Here is the matrix mat:
        // 1 2
        // 4 7

        // After 'mat = 2 * mat', mat =
        //  2  4
        //  8 14

        // After the subtraction, it becomes
        //  1  4
        //  8 13

        // After squaring, it becomes
        //   1  16
        //  64 169

        // Doing everything at once yields
        //   1  16
        //  64 169
        //通常，如果表达式右侧的（i，j）条目仅取决于左侧矩阵或数组的（i，j）条目
        //而不依赖于其他任何表达式，则赋值是安全的。在这种情况下，不必显示地评估右侧(.evl())。
}

void AliasingAndMatrixMultiplication()
{
        //在目标矩阵**未调整大小**的情况下，矩阵乘法是Eigen中唯一假定默认情况下为混淆的。若假定混淆，则会使用eval()生成临时对象,所以是安全的。因此，如果matA是平方矩阵，则该语句matA = matA * matA是安全的。Eigen中的所有其他操作都假定没有混淆问题，这是因为结果被分配给了不同的矩阵，或者因为它是逐个元素的操作。
        {
                MatrixXf matA(2, 2);
                matA << 2, 0, 0, 2;
                matA = matA * matA;
                cout << matA << endl;
        }

        // 但是，这是有代价的。执行表达式时matA = matA * matA, Eigen会在计算后的临时矩阵中评估赋值给matA的乘积。虽然可以，但是在将乘积分配给其他矩阵（例如matB = matA * matA）时，Eigen会执行相同的操作。在这种情况下，直接评估matB,而不是先将matA*matA生成临时对象，然后评估临时对象，最后将临时对象赋值给矩阵matB更高效. 我们可以使用noalias函数指示没有混淆，如下所示：matB.noalias() = matA * matA。这使Eigen可以matA * matA直接将矩阵乘积在matB中评估。
        {
                MatrixXf matA(2, 2), matB(2, 2);
                matA << 2, 0, 0, 2;
                // 简单但是效率低
                matB = matA * matA;
                cout << matB << endl;
                // 复杂但是效率高
                matB.noalias() = matA * matA;
                cout << matB << endl;
        }

        {
                //此外，从Eigen 3.3 开始，
                //如果调整了目标矩阵的大小(注意，上文中的操作假定目标矩阵大小不变)
                //并且未将乘积直接分配给目标，则不假定混淆。因此，以下示例也是错误的：
                MatrixXf A(2, 2), B(3, 2);
                B << 2, 0, 0, 3, 1, 1;
                A << 2, 0, 0, -2;
                A = (B * A).cwiseAbs(); // 由于不假定混淆，所以需要我们显示评价
                cout << A << endl;
        }
        {
                //对于任何混淆问题，您都可以通过在赋值之前显式评估表达式来解决它：
                MatrixXf A(2, 2), B(3, 2);
                B << 2, 0, 0, 3, 1, 1;
                A << 2, 0, 0, -2;
                A = (B * A).eval().cwiseAbs();
                cout << A << endl;

                // Output is
                // 4 0
                // 0 6
                // 2 2
        }
}


Section10_StorageOrders

//矩阵和二维数组有两种不同的存储顺序：列优先和行优先。本页说明了这些存储顺序以及如何指定应使用的存储顺序。矩阵的条目形成一个二维网格。但是，当矩阵存储在存储器中时，必须以某种方式线性排列条目。有两种主要方法可以做到这一点，按行和按列。我们说矩阵是**按行优先存储**。首先存储整个第一行，然后存储整个第二行，依此类推。另一方面，如果矩阵是逐列存储的，则以主列顺序存储，从整个第一列开始，然后是整个第二列，依此类推。

//可以通过Options为Matrix或Array指定模板参数来设置矩阵或二维数组的存储顺序。
//由于Matrix类解释，Matrix类模板有六个模板参数，
// 其中三个是强制性的（Scalar，RowsAtCompileTime和ColsAtCompileTime）三个是可选的（Options，MaxRowsAtCompileTime和MaxColsAtCompileTime）。如果Options参数设置为RowMajor，则矩阵或数组以行优先顺序存储；如果将其设置为ColMajor，则以列优先顺序存储。如果未指定存储顺序，则Eigen默认将条目存储在column-major中。如果方便的typedef（Matrix3f，ArrayXXd等）也是默认按列存储。

//可以将使用一种存储顺序的矩阵和数组分配给使用另一种存储顺序的矩阵和数组，如一个按行存储的矩阵使用按列存储矩阵初始化。
//Eigen将自动对元素重新排序。更一般而言，按行存储矩阵和按列存储矩阵可以根据需要在表达式中混合使用。

//当矩阵以行优先顺序存储时，逐行遍历矩阵的算法会更快，因为数据位置更好。同样，对于主要列矩阵，逐列遍历更快。可能需要尝试一下以找出对您的特定应用程序更快的方法。
//Eigen中的默认值是column-major。自然，对Eigen库的大多数开发和测试都是使用列主矩阵完成的。这意味着，即使我们旨在透明地支持列主存储和行主存储顺序，Eigen库也最好与列主存储矩阵配合使用。
void testColumnAndRowMajorStorage()
{
        // PlainObjectBase::data()  返回第一个元素的内存位置，和C++的数组名作用一样

        Matrix<int, 3, 4, ColMajor> Acolmajor;
        Acolmajor << 8, 2, 2, 9,
            9, 1, 4, 4,
            3, 5, 4, 5;
        cout << "The matrix A:" << endl;
        cout << Acolmajor << endl
             << endl;
        cout << "In memory (column-major):" << endl;
        for (int i = 0; i < Acolmajor.size(); i++)
                cout << *(Acolmajor.data() + i) << "  ";
        cout << endl
             << endl;
        Matrix<int, 3, 4, RowMajor> Arowmajor = Acolmajor;
        cout << "In memory (row-major):" << endl;
        for (int i = 0; i < Arowmajor.size(); i++)
                cout << *(Arowmajor.data() + i) << "  ";
        cout << endl;

        // In memory (column-major):
        // 8  9  3  2  1  5  2  4  4  9  4  5

        // In memory (row-major):
        // 8  2  2  9  9  1  4  4  3  5  4  5
}



对齐

namespace Section11_AlignmentIssues
{
// 对齐错误
//Eigen::internal::matrix_array<T, Size, MatrixOptions, Align>::internal::matrix_array()
// [with T = double, int Size = 2, int MatrixOptions = 2, bool Align = true]:
// Assertion `(reinterpret_cast<size_t>(array) & (sizemask)) == 0 && "this assertion
// is explained here: http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
//      READ THIS WEB PAGE !!! ****"' failed.

// 首先找到程序触发位置，
// 例如，
//$gdb ./my_program
//>run
//...
//>bt
//然后与下面的4种原因对号入座，修改代码

// 二 四种原因

// 原因1:结构体中具有Eigen对象成员
//请注意，此处Eigen :: Vector2d仅用作示例，
//更一般而言，所有固定大小的可矢量化Eigen类型都会出现此问题
//固定大小的可矢量化Eigen类型是如果它具有固定的大小并且大小是16字节的倍数
// Eigen::Vector2d
// Eigen::Vector4d
// Eigen::Vector4f
// Eigen::Matrix2d
// Eigen::Matrix2f
// Eigen::Matrix4d
// Eigen::Matrix4f
// Eigen::Affine3d
// Eigen::Affine3f
// Eigen::Quaterniond
// Eigen::Quaternionf

// 首先, "固定大小"应该清楚：如果在编译时，Eigen对象的行数和列数是固定的，则其固定大小。
// 因此，例如Matrix3f具有固定大小，但MatrixXf没有（固定大小的对立是动态大小）。
// 固定大小的Eigen对象的系数数组是普通的"静态数组"，不会动态分配。例如，Matrix4f后面的数据只是一个"float array[16]"。
// 固定大小的对象通常很小，这意味着我们要以零的运行时开销（在内存使用和速度方面）来处理它们。
// 现在，矢量化（SSE和AltiVec）都可以处理128位数据包。此外，出于性能原因，这些数据包必须具有128位对齐。因此，事实证明，固定大小的Eigen对象可以向量化的唯一方法是，如果它们的大小是128位或16个字节的倍数。然后，Eigen将为这些对象请求16字节对齐，并且此后将依赖于这些对象进行对齐，因此不会执行运行时检查以进行对齐。
// class Foo
// {
//   //...
//   Eigen::Vector2d v;
//   //...
// };
// //...
// Foo *foo = new Foo;

// Eigen需要Eigen :: Vector2d的数组（2个双精度）的128位对齐。对于GCC，这是通过属性（（aligned（16）））完成的。
// Eigen重载了Eigen :: Vector2d的" operator new"，因此它将始终返回128位对齐的指针。
// 因此，通常情况下，您不必担心任何事情，Eigen会为您处理对齐...

// ...除了一种情况。当您具有上述的Foo类，并且如上所述动态分配新的Foo时，则由于Foo没有对齐" operator new"，因此返回的指针foo不一定是128位对齐的。
// 然后，成员v的alignment属性相对于类的开头foo。如果foo指针未对齐，则foo-> v也不会对齐！
// 解决方案是让Foo类具有一致的"Operator new"

// 解决方法：
//如果定义的结构具有固定大小的可矢量化Eigen类型的成员，则必须重载其" operator new"，以便它生成16字节对齐的指针。幸运的是，Eigen为您提供了一个宏EIGEN_MAKE_ALIGNED_OPERATOR_NEW来为您执行此操作。换句话说：您有一个类，该类具有固定大小的可矢量化Eigen对象作为成员，然后动态创建该类的对象。很简单，您只需将EIGEN_MAKE_ALIGNED_OPERATOR_NEW宏放在您的类的public部分
// class Foo
// {
//         Eigen::Vector2d v;
//         public : EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// };
// Foo *foo = new Foo;
//该宏使" new Foo"始终返回对齐的指针。
//一个 Eigen::Vector2d有两个double类型，一个double为8字节=64位，则一个Eigen::Vector2d为128位
//这恰好是SSE数据包的大小，这使得可以使用SSE对该向量执行各种操作。但是SSE指令（至少Eigen使用的是快速指令）需要128位对齐。否则会出现段错误。出于这个原因，Eigen自己通过执行以下两项工作自行要求对Eigen :: Vector2d进行128位对齐：


//原因2：STL容器或手动内存分配
///如果您将Stl容器（例如std :: vector，std :: map等）
//与Eigen对象或包含Eigen对象的类一起使用，
// std::vector<Eigen::Matrix2f> my_vector;
// struct my_class { ... Eigen::Matrix2f m; ... };
// std::map<int, my_class> my_map;
//请注意，此处Eigen :: Matrix2f仅用作示例，更一般而言，对于所有固定大小的可矢量化Eigen类型和具有此类Eigen对象作为member的结构，都会出现此问题。任何类/函数都会绕过new分配内存的运算符，也就是通过执行自定义内存分配，然后调用placement new运算符，也会出现相同的问题。例如，通常就是这种情况，std::make_shared或者std::allocate_shared解决方案是使用对齐的分配器，如STL容器解决方案中所述。

//原因3：通过值传递Eigen对象
//如果您代码中的某些函数正在通过值传递Eigen对象，例如这样，
//void func(Eigen::Vector4d v);
//那么您需要阅读以下单独的页面：将Eigen对象按值传递给函数。请注意，此处Eigen :: Vector4d仅用作示例，更一般而言，所有固定大小的可矢量化Eigen类型都会出现此问题


// 3  该statement的一般解释:
// 固定大小的矢量化Eigen对象必须绝对在16字节对齐的位置创建，否则寻址它们的SIMD指令将崩溃。Eigen通常会为您解决这些对齐问题，方法是在固定大小的可矢量化对象上设置对齐属性，并重载他们的" operator new"。但是，在某些极端情况下，这些对齐设置会被覆盖：它们是此断言的可能原因。

}



Section1_LinearAlgebraAndDecompositions
{
//本页说明了如何求解线性系统，计算各种分解，例如LU，QR，SVD，本征分解...
//基本线性求解 Ax=b
void BasicLinearSolving()
{
        Matrix3f A;
        Vector3f b;
        A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
        b << 3, 3, 4;
        cout << "Here is the matrix A:\n"
             << A << endl;
        cout << "Here is the vector b:\n"
             << b << endl;
        Vector3f x = A.colPivHouseholderQr().solve(b);
        cout << "The solution is:\n"
             << x << endl;

        //在此示例中，colPivHouseholderQr（）方法返回类ColPivHouseholderQR的对象。
        //由于此处的矩阵类型为Matrix3f，因此该行可能已替换为：ColPivHouseholderQR <Matrix3f> dec（A）;
        //Vector3f x = dec.solve（b）;在这里，ColPivHouseholderQR是具有选择列主元功能的QR分解。这是本教程的一个不错的折衷方案，因为它适用于所有矩阵，而且速度非常快。这是一些其他分解表，您可以根据矩阵和要进行的权衡选择：
        {
                //所有这些分解都提供了一个solve（）方法，该方法与上述示例一样。
                //例如，如果您的矩阵是正定的，则上表说明LLT或LDLT分解是一个很好的选择。这是一个示例，也说明使用通用矩阵（而非矢量）作为右手边是可能的。
                Matrix2f A, b;
                A << 2, -1, -1, 3;
                b << 1, 2, 3, 1;
                cout << "Here is the matrix A:\n"
                     << A << endl;
                cout << "Here is the right hand side b:\n"
                     << b << endl;
                Matrix2f x = A.ldlt().solve(b);
                cout << "The solution is:\n"
                     << x << endl;
        }
}

void CheckingIfASolutionReallyExists()
{
        //计算相对误差的方法
        //只有您知道要允许解决方案被视为有效的误差范围。因此，Eigen允许您自己进行此计算，如以下示例所示：
        MatrixXd A = MatrixXd::Random(100, 100);
        MatrixXd b = MatrixXd::Random(100, 50);
        MatrixXd x = A.fullPivLu().solve(b);
        double relative_error = (A * x - b).norm() / b.norm(); // norm() is L2 norm
        cout << "The relative error is:\n"
             << relative_error << endl;
}

void ComputingEigenvaluesAndEigenvectors()
{
        //您需要在此处进行特征分解， 确保检查矩阵是否是自伴随的，在数学里，作用于一个有限维的内积空间，一个自伴算子(self-adjoint operator)等于自己的伴随算子；等价地说，表达自伴算子的矩阵是埃尔米特矩阵。埃尔米特矩阵等于自己的共轭转置。根据有限维的谱定理，必定存在着一个正交归一基，可以表达自伴算子为一个实值的对角矩阵。就像在这些问题中经常发生的那样。这是一个使用SelfAdjointEigenSolver的示例，可以使用EigenSolver或ComplexEigenSolver轻松地将其应用于一般矩阵。特征值和特征向量的计算不一定会收敛，但是这种收敛失败的情况很少。调用info（）就是为了检查这种可能性。
        Matrix2f A;
        A << 1, 2, 2, 3;
        SelfAdjointEigenSolver<Matrix2f> eigensolver(A);
        if (eigensolver.info() != Success)
                abort();
        cout << "The eigenvalues of A are:\n"
             << eigensolver.eigenvalues() << endl;
        cout << "Here's a matrix whose columns are eigenvectors of A \n"
             << "corresponding to these eigenvalues:\n"
             << eigensolver.eigenvectors() << endl;
}

void ComputingInverseAndDeterminant()
{
        //逆计算通常可以用solve（）操作代替，而行列式通常不是检查矩阵是否可逆的好方法。但是，对于非常小的矩阵，上述条件是不正确的，并且逆和行列式可能非常有用。尽管某些分解（例如PartialPivLU和FullPivLU）提供了inverse（）和determinant（）方法，但您也可以直接在矩阵上调用inverse（）和determinant（）。如果矩阵的固定大小很小（最多为4x4），Eigen应避免执行LU分解，而应使用对此类小矩阵更有效的公式。
        Matrix3f A;
        A << 1, 2, 1,
            2, 1, 0,
            -1, 1, 2;
        cout << "Here is the matrix A:\n"
             << A << endl;
        cout << "The determinant of A is " << A.determinant() << endl;   //行列式
        cout << "The inverse of A is:\n"
             << A.inverse() << endl;
}

void LeastSquaresSolving()
{
        //最小二乘求解的最准确方法是SVD分解。Eigen提供了两种实现。推荐的对象是BDCSVD类，它可以很好地解决较大的问题，并自动退回到JacobiSVD类以解决较小的问题。对于这两个类，它们的resolve（）方法都在进行最小二乘求解。
        MatrixXf A = MatrixXf::Random(3, 2);
        cout << "Here is the matrix A:\n"
             << A << endl;
        VectorXf b = VectorXf::Random(3);
        cout << "Here is the right hand side b:\n"
             << b << endl;
        cout << "The least-squares solution is:\n"
             << A.bdcSvd(ComputeThinU | ComputeThinV).solve(b) << endl;

        //可能更快但可靠性较低的另一种方法是使用矩阵的Cholesky分解或QR分解。我们关于最小二乘法求解的页面有更多详细信息。
}

void SeparatingTheComputationFromTheConstruction()
{
        // 在以上示例中，在构造分解对象的同时计算了分解。但是，在某些情况下，您可能希望将这两件事分开，
        //例如，如果在构造时不知道要分解的矩阵，则可能会需要将它们分开。或者您想重用现有的分解对象。
        // 使之成为可能的原因是：
        // 所有分解都有默认的构造函数，
        // 所有分解都具有执行计算的compute（matrix）方法，并且可以在已计算的分解中再次调用该方法，以将其重新初始化。
        Matrix2f A, b;
        LLT<Matrix2f> llt;
        A << 2, -1, -1, 3;
        b << 1, 2, 3, 1;
        cout << "Here is the matrix A:\n"
             << A << endl;
        cout << "Here is the right hand side b:\n"
             << b << endl;
        cout << "Computing LLT decomposition..." << endl;
        llt.compute(A);
        cout << "The solution is:\n"
             << llt.solve(b) << endl;
        //最后，您可以告诉分解构造函数预先分配存储空间以分解给定大小的矩阵，以便在随后分解此类矩阵时，不执行动态内存分配（当然，如果您使用的是固定大小的矩阵，则不存在动态内存分配完全发生）。只需将大小传递给分解构造函数即可完成，如以下示例所示：
        {
                HouseholderQR<MatrixXf> qr(50, 50);
                MatrixXf A = MatrixXf ::Random(50, 50);
                qr.compute(A); //没有动态内存分配
        }
}
void RankRevealingDecompositions()
{
        //某些分解是揭示矩阵秩的。
        //这些通常也是在非满秩矩阵（在方形情况下表示奇异矩阵）的情况下表现最佳的分解。
        //秩揭示分解至少提供了rank()方法。它们还可以提供方便的方法，例如isInvertible()，
        //并且还提供一些方法来计算矩阵的核（零空间）和像（列空间），就像FullPivLU那样：
        Matrix3f A;
        A << 1, 2, 5,
            2, 1, 4,
            3, 0, 3;
        cout << "Here is the matrix A:\n"
             << A << endl;
        FullPivLU<Matrix3f> lu_decomp(A);
        cout << "The rank of A is " << lu_decomp.rank() << endl;
        cout << "Here is a matrix whose columns form a basis of the null-space of A:\n"
             << lu_decomp.kernel() << endl;
        cout << "Here is a matrix whose columns form a basis of the column-space of A:\n"
             << lu_decomp.image(A) << endl; // yes, have to pass the original A

        // Output is:
        // Here is the matrix A:
        // 1 2 5
        // 2 1 4
        // 3 0 3
        // The rank of A is 2
        // Here is a matrix whose columns form a basis of the null-space of A:
        //  0.5
        //    1
        // -0.5
        // Here is a matrix whose columns form a basis of the column-space of A:
        // 5 1
        // 4 2
        // 3 3

        //当然，任何秩计算都取决于对任意阈值的选择，因为实际上没有浮点矩阵恰好是秩不足的。Eigen选择一个明智的默认阈值，该阈值取决于分解，但通常是对角线大小乘以机器ε。虽然这是我们可以选择的最佳默认值，但只有您知道您的应用程序的正确阈值是多少。您可以通过在调用rank（）或需要使用此阈值的任何其他方法之前在分解对象上调用setThreshold（）来进行设置。分解本身（即compute（）方法）与阈值无关。更改阈值后，无需重新计算分解.
        {
                Matrix2d A;
                A << 2, 1,
                    2, 0.9999999999;
                FullPivLU<Matrix2d> lu(A);
                cout << "By default, the rank of A is found to be " << lu.rank() << endl;
                lu.setThreshold(1e-5);
                cout << "With threshold 1e-5, the rank of A is found to be " << lu.rank() << endl;

                // Output is:
                // By default, the rank of A is found to be 2
                // With threshold 1e-5, the rank of A is found to be 1
        }
}

Section2_CatalogueOfDenseDecompositions
{
// 此页面显示了Eigen提供的稠密矩阵分解的目录。（被我删掉了 自己找）

// 笔记：
// 1：LDLT算法有两种变体。Eigen的一个生成纯对角D矩阵，因此它不能处理不定的矩阵，这与Lapack的一个生成块对角D矩阵不同。
// 2：特征值，SVD和Schur分解依赖于迭代算法。它们的收敛速度取决于特征值的分离程度。
// 3：我们的JacobiSVD是双面的，可为平方矩阵提供经过验证的最佳精度。对于非平方矩阵，我们必须首先使用QR预调节器。默认选择ColPivHouseholderQR已经非常可靠，但是如果您想证明这一点，请改用FullPivHouseholderQR。
// 术语：
// 自伴：对于实矩阵，自伴是对称的同义词。对于复杂的矩阵，自伴为同义词埃尔米特。更一般地，A当且仅当矩阵等于其伴随时A*，矩阵才是自伴随的$ A ^ * $。伴随也称为共轭转置。
// Blocking
// 意味着该算法可以按块工作，从而保证了大型矩阵性能的良好缩放。
// 隐式多线程（MT）
// 意味着该算法可以通过OpenMP利用多核处理器。"隐式"是指算法本身不是并行的，而是依赖于并行矩阵矩阵乘积的规则。
// 显式多线程（MT）
// 意味着该算法已显式并行化，以通过OpenMP利用多核处理器。
// 元展开器
// 意味着对于很小的固定大小矩阵，该算法将自动显式展开。
} 

