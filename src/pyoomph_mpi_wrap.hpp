#pragma once

#include <cstdint>

// See https://stackoverflow.com/questions/38442254/how-to-write-an-mpi-wrapper-for-dynamic-loading
typedef intptr_t pyoomph_MPI_Datatype;
typedef intptr_t pyoomph_MPI_Comm;
// TODO: Check these!
typedef int pyoomph_MPI_Op;
typedef int pyoomph_MPI_Group;

#define MPI_Datatype pyoomph_MPI_Datatype
#define MPI_Comm     pyoomph_MPI_Comm
#define MPI_Op     pyoomph_MPI_Op
#define MPI_Group     pyoomph_MPI_Group

#define MPI_COMM_WORLD pyoomph_MPI_COMM_WORLD
#define MPI_INT        pyoomph_MPI_INT
#define MPI_DOUBLE        pyoomph_MPI_DOUBLE
#define MPI_UNSIGNED pyoomph_MPI_UNSIGNED
#define MPI_UNSIGNED_LONG pyoomph_MPI_UNSIGNED_LONG
#define MPI_IDENT        pyoomph_MPI_IDENT
#define MPI_SUM        pyoomph_MPI_SUM
#define MPI_MAX        pyoomph_MPI_MAX


// Should be sufficient, hopefully...
#define PYOOMPH_MAX_MPI_STATUS_SIZE 64

typedef struct pyoomph_MPI_Status
{
   int MPI_SOURCE;
   int MPI_TAG;
   int MPI_ERROR;
   char _original[PYOOMPH_MAX_MPI_STATUS_SIZE];
} pyoomph_MPI_Status;

#define MPI_Status        pyoomph_MPI_Status
#define MPI_STATUS_IGNORE ((pyoomph_MPI_Status*)NULL)


// Required types to be set dynamically
extern MPI_Comm     pyoomph_MPI_COMM_WORLD;
extern MPI_Datatype pyoomph_MPI_INT;
extern MPI_Datatype pyoomph_MPI_DOUBLE;
extern MPI_Datatype pyoomph_MPI_UNSIGNED;
extern MPI_Datatype pyoomph_MPI_UNSIGNED_LONG;
extern pyoomph_MPI_Group pyoomph_MPI_IDENT;
extern pyoomph_MPI_Op pyoomph_MPI_SUM;
extern pyoomph_MPI_Op pyoomph_MPI_MAX;

extern int (*pyoomph_MPI_Init)(int *,char ***) ;
extern int (*pyoomph_MPI_Comm_free)(MPI_Comm *) ;
extern int (*pyoomph_MPI_Comm_size)(MPI_Comm ,int*) ;
extern int (*pyoomph_MPI_Comm_rank)(MPI_Comm ,int*) ;
extern int (*pyoomph_MPI_Comm_compare)(MPI_Comm , MPI_Comm, int *);
extern int (*pyoomph_MPI_Comm_split)(MPI_Comm, int, int, MPI_Comm *);
extern int (*pyoomph_MPI_Bcast)(void *, int , MPI_Datatype , int ,MPI_Comm);
extern int (*pyoomph_MPI_Allreduce)(const void *, void *, int,MPI_Datatype , MPI_Op , MPI_Comm );
extern int (*pyoomph_MPI_Sendrecv)(const void *, int, MPI_Datatype,int, int,void *, int, MPI_Datatype,int, int,MPI_Comm, MPI_Status *);
extern int (*pyoomph_MPI_Allgatherv)(const void *, int , MPI_Datatype ,void *, const int[], const int[],MPI_Datatype, MPI_Comm);
extern int (*pyoomph_MPI_Alltoall)(const void *, int, MPI_Datatype,void *, int , MPI_Datatype ,MPI_Comm );
extern int (*pyoomph_MPI_Alltoallv)(const void *, const int [],const int [], MPI_Datatype , void *,const int [], const int [],MPI_Datatype,MPI_Comm);
extern int (*pyoomph_MPI_Send)(const void *, int , MPI_Datatype , int,int, MPI_Comm);
extern int (*pyoomph_MPI_Recv)(void *, int, MPI_Datatype, int, int,MPI_Comm, MPI_Status *);
extern int (*pyoomph_MPI_Allgather)(const void *, int, MPI_Datatype,void *, int, MPI_Datatype ,MPI_Comm);

static int MPI_Init(int *argc, char ***argv)
{
   return  pyoomph_MPI_Init(argc, argv);
}

static int MPI_Comm_free(MPI_Comm * comm)
{
   return  pyoomph_MPI_Comm_free(comm);
}

static int MPI_Comm_size( MPI_Comm comm, int *size ) 
{
 return pyoomph_MPI_Comm_size(comm,size);
}

static int MPI_Comm_rank( MPI_Comm comm, int *rank ) 
{
 return pyoomph_MPI_Comm_rank(comm,rank);
}

static int MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result)
{
 return pyoomph_MPI_Comm_compare(comm1,comm2,result);
}


static int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
 return pyoomph_MPI_Comm_split(comm,color, key,newcomm);
}

static int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,MPI_Comm comm)
{
 return pyoomph_MPI_Bcast(buffer,count,datatype,root,comm);
}

static int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
 return pyoomph_MPI_Allreduce(sendbuf, recvbuf,  count, datatype, op, comm);
}


static int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,int dest, int sendtag,void *recvbuf, int recvcount, MPI_Datatype recvtype,int source, int recvtag,MPI_Comm comm, MPI_Status *status)
{
 return pyoomph_MPI_Sendrecv(sendbuf, sendcount,sendtype, dest, sendtag,recvbuf, recvcount,recvtype,source,recvtag,comm,status);
}

static int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,void *recvbuf, const int recvcounts[], const int displs[],MPI_Datatype recvtype, MPI_Comm comm)
{
 return pyoomph_MPI_Allgatherv(sendbuf, sendcount, sendtype,recvbuf,recvcounts,displs,recvtype, comm);
}

static int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,void *recvbuf, int recvcount, MPI_Datatype recvtype,MPI_Comm comm)
{
 return pyoomph_MPI_Alltoall(sendbuf, sendcount,sendtype,recvbuf, recvcount,recvtype,comm);
}

static int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],const int sdispls[], MPI_Datatype sendtype, void *recvbuf,const int recvcounts[], const int rdispls[],MPI_Datatype recvtype, MPI_Comm comm)
{
 return pyoomph_MPI_Alltoallv(sendbuf,sendcounts,sdispls,sendtype,recvbuf,recvcounts,rdispls, recvtype, comm);
}

static int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,int tag, MPI_Comm comm)
{
 return pyoomph_MPI_Send(buf, count, datatype, dest,tag,  comm);
}

static int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,MPI_Comm comm, MPI_Status *status)
{
 return pyoomph_MPI_Recv(buf, count,datatype, source, tag, comm,status);
}

static int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,void *recvbuf, int recvcount, MPI_Datatype recvtype,MPI_Comm comm)
{
 return pyoomph_MPI_Allgather(sendbuf, sendcount, sendtype,recvbuf, recvcount, recvtype, comm);
}
