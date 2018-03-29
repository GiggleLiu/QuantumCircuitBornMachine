#================================================
# system parameters
MODULE = fysics
CPL = f2py
LIBS = -llapack #-lmkl_intel -lmkl_sequential -lmkl_core -llapack
SOURCES = fysics.f90
OBJECTS=$(SOURCES:.f90=.o)
OPT = --overwrite-signature
#================================================
# link all to generate exe file
#$(MODULE).so: $(OBJS)
	#$(CPL) $(FCOPTS) $(OBJS) $(LIBS) -o main.out 
#================================================
#generate every obj and module files

all:$(MODULE).so
$(MODULE).so : $(SOURCES)
	$(CPL) -m $(MODULE) -c $(SOURCES) $(LIBS)

#$(MODULE).so : $(OBJECTS)
	#$(CPL) $(LIBS) -c $(MODULE).pyf $(OBJECTS)
#
#$(OBJECTS) $(MODULE).pyf : $(SOURCES)
	#$(CPL) -m $(MODULE) -h $(MODULE).pyf $(SOURCES) $(OPT)
#================================================
clean:
	rm -f *.so *.o *.pyh
