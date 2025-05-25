###############################################################################
# 編譯器與參數設定
###############################################################################
NVCC       := nvcc
GXX        := g++
CFLAGS     := -O3 -g
Capability := 60
ARCH       := -arch sm_$(Capability)

INCLUDES_MPI   := -I /opt/openmpi-3.1.4/include/
LIBS_MPI       := -L /opt/openmpi-3.1.4/lib/ -lmpi
INCLUDES_EIGEN := -I /home/jeff/eigen/eigen-3.3.9

###############################################################################
# 目標檔案與輸出
###############################################################################
# 輸出執行檔
TARGET := a.out

# 產生 .o 檔要放的資料夾
OBJDIR := ./OBJ

# 資料輸出資料夾
DATADIR := ./data

# 原始碼: 需要用 nvcc 編譯的 (CUDA)
SRCS_CU := main.cu

# 原始碼: 需要用 g++ 編譯的 (含 Eigen)
SRCS_CPP := linear_algebra.cpp

# 依檔名規則生成對應的 .o
OBJS_CU  := $(SRCS_CU:%.cu=$(OBJDIR)/%.o)
OBJS_CPP := $(SRCS_CPP:%.cpp=$(OBJDIR)/%.o)
OBJS     := $(OBJS_CU) $(OBJS_CPP)

###############################################################################
# 編譯規則
###############################################################################
# 若 OBJ 資料夾尚未存在，建立它
$(shell mkdir -p $(OBJDIR))

# 若 data 資料夾尚未存在，建立它
$(shell mkdir -p $(DATADIR))

# (1) 編譯 .cu 檔 -> .o
$(OBJDIR)/%.o: %.cu
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES_MPI) $(INCLUDES_EIGEN) -c $< -o $@

# (2) 編譯 .cpp 檔 -> .o
$(OBJDIR)/%.o: %.cpp
	$(GXX)  $(CFLAGS) $(INCLUDES_MPI) $(INCLUDES_EIGEN) -c $< -o $@

###############################################################################
# 連結
###############################################################################
# 最終產生 a.out
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(ARCH) $(INCLUDES_MPI) $(INCLUDES_EIGEN) $(LIBS_MPI) $(OBJS) -o $@

###############################################################################
# 清除
###############################################################################
.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)
